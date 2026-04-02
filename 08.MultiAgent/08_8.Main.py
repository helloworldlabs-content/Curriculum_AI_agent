import importlib.util
import logging
import os
import sys

from fastapi import Depends, FastAPI, HTTPException

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.main")


def _load(module_name: str, filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_schemas      = _load("schemas_08_main",      "08_2.AgentSchemas.py")
_auth         = _load("auth_08",              "07_5.Auth.py")
_helpers      = _load("agent_helpers_08_main","08_3.AgentHelpers.py")
_orchestrator = _load("orchestrator_08",      "08_7.Orchestrator.py")

Message           = _schemas.Message
ChatRequest       = _schemas.ChatRequest
ChatResponse      = _schemas.ChatResponse
OrchestratorState = _schemas.OrchestratorState
LoginRequest      = _schemas.LoginRequest
TokenResponse     = _schemas.TokenResponse

create_token  = _auth.create_token
verify_token  = _auth.verify_token
authenticate  = _auth.authenticate

run_orchestrator   = _orchestrator.run_orchestrator
setup_vector_store = _helpers.setup_vector_store

# 앱 시작 시 벡터스토어 초기화 (한 번만)
_vectorstore = setup_vector_store()

app = FastAPI(title="Multi Agent Curriculum Backend")


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
    클라이언트가 전체 대화 기록과 오케스트레이터 상태를 보내면
    현재 phase에 맞는 에이전트를 실행하고 결과를 반환한다.

    응답:
    - reply: 에이전트의 텍스트 응답
    - complete: 커리큘럼 생성 완료 여부
    - curriculum: 생성된 커리큘럼 (complete=True일 때만 존재)
    - state: 다음 요청 시 전달할 오케스트레이터 상태
    - active_agent: 이번 턴에 실행된 에이전트 이름
    """
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        state_dict = (req.state or OrchestratorState()).model_dump()

        result = run_orchestrator(messages, state_dict, _vectorstore)

        return ChatResponse(
            reply=result["reply"],
            complete=result["complete"],
            curriculum=result.get("curriculum"),
            state=OrchestratorState.model_validate(result["state"]),
            active_agent=result["active_agent"],
        )
    except Exception as error:
        logger.exception("[chat] error: %s", error)
        raise HTTPException(status_code=500, detail=str(error))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
