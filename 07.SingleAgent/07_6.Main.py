import importlib.util
import logging
import os
import sys

from fastapi import Depends, FastAPI, HTTPException

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("single_agent.main")


def _load(module_name: str, filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_schemas = _load("schemas",     "07_2.AgentSchemas.py")
_auth    = _load("auth",        "07_5.Auth.py")
_agent   = _load("agent",       "07_4.SingleAgent.py")
_helpers = _load("agent_helpers", "07_3.AgentHelpers.py")

Message       = _schemas.Message
ChatRequest   = _schemas.ChatRequest
ChatResponse  = _schemas.ChatResponse
LoginRequest  = _schemas.LoginRequest
TokenResponse = _schemas.TokenResponse

create_token    = _auth.create_token
verify_token    = _auth.verify_token
authenticate    = _auth.authenticate

run_agent         = _agent.run_agent
setup_vector_store = _helpers.setup_vector_store

# 앱 시작 시 벡터스토어 초기화 (한 번만)
_vectorstore = setup_vector_store()

app = FastAPI(title="Single Agent Curriculum Backend")


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
    """
    클라이언트가 전체 대화 기록을 보내면 에이전트 루프를 실행하고 결과를 반환한다.

    응답:
    - reply: 에이전트의 텍스트 응답
    - complete: 커리큘럼 생성 완료 여부
    - curriculum: 생성된 커리큘럼 (complete=True일 때만 존재)
    """
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        result = run_agent(messages, _vectorstore)
        return ChatResponse(
            reply=result["reply"],
            complete=result["complete"],
            curriculum=result.get("curriculum"),
            validation_result=result.get("validation_result"),
        )
    except Exception as error:
        logger.exception("[chat] error: %s", error)
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
