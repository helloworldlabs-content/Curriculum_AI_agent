# 빌드 컨텍스트: 프로젝트 루트 (Curriculum_AI_agent/)
# 실행: docker build -f 05.Streamlit/05-6.Dockerfile -t curriculum-backend .

FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사 (파일명 하이픈으로 인해 importlib 사용 — 원본 파일명 유지)
# 05-5.main.py는 gunicorn 모듈명 호환을 위해 main.py로 복사
COPY 05.Streamlit/05-2.schemas.py ./05-2.schemas.py
COPY 05.Streamlit/05-3.auth.py    ./05-3.auth.py
COPY 05.Streamlit/05-4.rag.py     ./05-4.rag.py
COPY 05.Streamlit/05-5.main.py    ./main.py

# PDF 및 캐시 파일 복사
COPY Data/ ./Data/

# vectorDB는 볼륨으로 마운트 (05-7.docker-compose.yml 참고)
RUN mkdir -p ./vectorDB

ENV APP_BASE_DIR=/app

EXPOSE 8000

# Gunicorn + uvicorn worker 설정
# --preload: 마스터에서 앱 로드 후 포크 → VectorDB 초기화 1회만 실행
# --timeout 300: LLM 생성 시간 고려
# --workers: CPU * 2 + 1 공식 기준 (조정 가능)
CMD ["gunicorn", "main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--preload", \
     "--timeout", "300"]
