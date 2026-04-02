import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.main")

# JSON 파일 저장 디렉터리
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _load(module_name: str, filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_schemas      = _load("schemas_08_main",       "08_2.AgentSchemas.py")
_auth         = _load("auth_08",               "07_5.Auth.py")
_helpers      = _load("agent_helpers_08_main", "08_3.AgentHelpers.py")
_orchestrator = _load("orchestrator_08",       "08_7.Orchestrator.py")

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

_vectorstore = setup_vector_store()

app = FastAPI(title="Multi Agent Curriculum Backend")


# ---------------------------------------------------------------------------
# JSON 파일 저장 헬퍼
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    """파일명에 사용할 수 없는 문자를 제거한다."""
    return re.sub(r"[^\w가-힣]", "_", text or "curriculum")[:40]


def _save_curriculum_json(curriculum: dict) -> str:
    """
    커리큘럼을 outputs/ 디렉터리에 JSON 파일로 저장하고 파일명을 반환한다.
    저장 실패 시 빈 문자열을 반환한다 (기능에는 영향 없음).
    """
    try:
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        title     = _safe_filename(curriculum.get("program_title", ""))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"{timestamp}_{title}.json"
        filepath  = os.path.join(_OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(curriculum, f, ensure_ascii=False, indent=2)
        logger.info("[main] curriculum saved: %s", filepath)
        return filename
    except Exception as err:
        logger.warning("[main] curriculum save failed: %s", err)
        return ""


# ---------------------------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------------------------

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
    커리큘럼이 완성된 경우 outputs/ 디렉터리에 JSON 파일로 저장한다.
    """
    try:
        messages   = [{"role": m.role, "content": m.content} for m in req.messages]
        state_dict = (req.state or OrchestratorState()).model_dump()

        result = run_orchestrator(messages, state_dict, _vectorstore)

        # 완성된 커리큘럼을 JSON 파일로 저장
        if result.get("complete") and result.get("curriculum"):
            _save_curriculum_json(result["curriculum"])

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


@app.get("/curriculum/files")
def list_curriculum_files(_: str = Depends(verify_token)) -> list[str]:
    """저장된 커리큘럼 JSON 파일 목록을 반환한다."""
    try:
        if not os.path.isdir(_OUTPUT_DIR):
            return []
        return sorted(
            [f for f in os.listdir(_OUTPUT_DIR) if f.endswith(".json")],
            reverse=True,
        )
    except Exception:
        return []


@app.get("/curriculum/download/{filename}")
def download_curriculum(filename: str, _: str = Depends(verify_token)):
    """저장된 커리큘럼 JSON 파일을 다운로드한다."""
    # path traversal 방지
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")
    filepath = os.path.join(_OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(filepath, media_type="application/json", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
