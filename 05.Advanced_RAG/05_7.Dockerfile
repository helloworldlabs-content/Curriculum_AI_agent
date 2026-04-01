# 빌드 컨텍스트: 프로젝트 루트 (Curriculum_AI_agent/)
# 실행: docker build -f 05.Advanced_RAG/05_7.Dockerfile -t curriculum-backend .

FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사 (파일명 하이픈으로 인해 importlib 사용 — 원본 파일명 유지)
# 05_6.Main.py는 gunicorn 모듈명 호환을 위해 main.py로 복사
COPY 05.Advanced_RAG/05_2.Schemas.py ./05_2.Schemas.py
COPY 05.Advanced_RAG/05_3.Auth.py    ./05_3.Auth.py
COPY 05.Advanced_RAG/05_4.Indexing.py ./05_4.Indexing.py
COPY 05.Advanced_RAG/05_5.Retrieval.py ./05_5.Retrieval.py
COPY 05.Advanced_RAG/05_6.Main.py     ./main.py

# PDF 및 캐시 파일 복사
COPY Data/ ./Data/

# vectorDB는 볼륨으로 마운트 (05_8.docker-compose.yml 참고)
RUN mkdir -p ./vectorDB

ENV APP_BASE_DIR=/app

EXPOSE 8000

# ChromaDB 임베디드 모드는 SQLite 기반으로 멀티프로세스 동시 접근을 지원하지 않는다.
# gunicorn 멀티 워커 사용 시 ChromaDB lock 충돌로 retrieval이 멈추는 문제가 발생한다.
# uvicorn 단일 프로세스로 실행하면 async 처리로 동시 요청을 처리하면서 문제를 회피한다.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "300"]
