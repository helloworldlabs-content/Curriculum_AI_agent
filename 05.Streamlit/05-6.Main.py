import importlib.util
import os
import sys

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _load(module_name: str, filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_schemas = _load("schemas", "05-2.Schemas.py")
_auth = _load("auth", "05-3.Auth.py")
_indexing = _load("indexing", "05-4.Indexing.py")
_retrieval = _load("retrieval", "05-5.Retrieval.py")


Message = _schemas.Message
ChatRequest = _schemas.ChatRequest
ChatResponse = _schemas.ChatResponse
GenerateRequest = _schemas.GenerateRequest
CollectedInfo = _schemas.CollectedInfo
CurriculumPlan = _schemas.CurriculumPlan
LoginRequest = _schemas.LoginRequest
TokenResponse = _schemas.TokenResponse

create_token = _auth.create_token
verify_token = _auth.verify_token
authenticate = _auth.authenticate

COLLECTION_SYSTEM_PROMPT = _retrieval.COLLECTION_SYSTEM_PROMPT
to_lc_messages = _retrieval.to_lc_messages
build_chain = _retrieval.build_chain
setup_vector_stores = _indexing.setup_vector_stores


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
USE_CONTEXTUAL_RETRIEVAL = os.getenv("USE_CONTEXTUAL_RETRIEVAL", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
_vectorstores = setup_vector_stores(contextual=USE_CONTEXTUAL_RETRIEVAL)
_chain = build_chain(_vectorstores)

app = FastAPI(title="AI Curriculum Backend")


def _build_groups(info: CollectedInfo) -> dict:
    return {
        "group_a": {
            "name": "그룹 A",
            "types": ["균형형", "이해형"],
            "count": info.count_balanced + info.count_learner,
        },
        "group_b": {
            "name": "그룹 B",
            "types": ["과신형", "실행형"],
            "count": info.count_overconfident + info.count_doer,
        },
        "group_c": {
            "name": "그룹 C",
            "types": ["판단형", "조심형"],
            "count": info.count_analyst + info.count_cautious,
        },
    }


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if not authenticate(req.username, req.password):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return TokenResponse(access_token=create_token(req.username))


@app.get("/health")
def health(_: str = Depends(verify_token)):
    collections = {name: store._collection.count() for name, store in _vectorstores.items()}
    return {"status": "ok", "chunks": sum(collections.values()), "collections": collections}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: str = Depends(verify_token)):
    try:
        lc_messages = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        response = _llm.invoke(lc_messages)
        reply = response.content
        complete = "[정보 수집 완료]" in reply

        collected_info = None
        if complete:
            extract_llm = _llm.with_structured_output(CollectedInfo)
            info: CollectedInfo = extract_llm.invoke(
                lc_messages
                + [AIMessage(content=reply)]
                + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
            )
            collected_info = info.model_dump()

        return ChatResponse(reply=reply, complete=complete, collected_info=collected_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate(req: GenerateRequest, _: str = Depends(verify_token)):
    try:
        info = CollectedInfo(**req.collected_info)
        conversation = [SystemMessage(content=COLLECTION_SYSTEM_PROMPT)] + to_lc_messages(req.messages)
        result: CurriculumPlan = _chain.invoke(
            {
                "conversation": conversation,
                "info": info,
                "groups": _build_groups(info),
            }
        )
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
