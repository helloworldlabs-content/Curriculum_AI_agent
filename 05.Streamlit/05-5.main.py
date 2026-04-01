import importlib.util
import os
import sys

# 파일명에 하이픈이 포함되어 일반 import가 불가능하므로 importlib로 로드한다.
def _load(filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[filename] = mod
    spec.loader.exec_module(mod)
    return mod

_schemas = _load("05-2.schemas.py")
_auth    = _load("05-3.auth.py")
_rag     = _load("05-4.rag.py")

# schemas
Message          = _schemas.Message
ChatRequest      = _schemas.ChatRequest
ChatResponse     = _schemas.ChatResponse
GenerateRequest  = _schemas.GenerateRequest
CollectedInfo    = _schemas.CollectedInfo
CurriculumPlan   = _schemas.CurriculumPlan
LoginRequest     = _schemas.LoginRequest
TokenResponse    = _schemas.TokenResponse

# auth
create_token  = _auth.create_token
verify_token  = _auth.verify_token
authenticate  = _auth.authenticate

# rag
COLLECTION_SYSTEM_PROMPT = _rag.COLLECTION_SYSTEM_PROMPT
to_lc_messages           = _rag.to_lc_messages
setup_vector_store       = _rag.setup_vector_store
build_chain              = _rag.build_chain

# ---

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os

_llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
_vectorstore = setup_vector_store()
_chain       = build_chain(_vectorstore)

app = FastAPI(title="AI 커리큘럼 백엔드")


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return TokenResponse(access_token=create_token(req.username))


@app.get("/health")
def health(_: str = Depends(verify_token)):
    return {"status": "ok", "chunks": _vectorstore._collection.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: str = Depends(verify_token)):
    """정보 수집 대화. complete=True 시 collected_info도 함께 반환한다."""
    try:
        lc_messages = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        response    = _llm.invoke(lc_messages)
        reply       = response.content
        complete    = "[정보 수집 완료]" in reply

        collected_info = None
        if complete:
            extract_llm = _llm.with_structured_output(CollectedInfo)
            info: CollectedInfo = extract_llm.invoke(
                lc_messages + [AIMessage(content=reply)]
                + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
            )
            collected_info = info.model_dump()

        return ChatResponse(reply=reply, complete=complete, collected_info=collected_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate(req: GenerateRequest, _: str = Depends(verify_token)):
    """collected_info를 받아 RAG 검색 + 커리큘럼 생성 (LLM 1회)."""
    try:
        info   = CollectedInfo(**req.collected_info)
        groups = {
            "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"],  "count": info.count_balanced + info.count_learner},
            "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"],  "count": info.count_overconfident + info.count_doer},
            "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"], "count": info.count_analyst + info.count_cautious},
        }
        conversation = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        result: CurriculumPlan = _chain.invoke({
            "conversation": conversation,
            "info":         info,
            "groups":       groups,
        })
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("05-2.main:app", host="0.0.0.0", port=8000, reload=False)
