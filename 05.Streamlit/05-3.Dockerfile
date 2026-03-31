# 빌드 컨텍스트: 프로젝트 루트 (Curriculum_AI_agent/)
# 실행: docker build -f 05.Streamlit/05-3.Dockerfile -t curriculum-backend .

FROM python:3.11-slim

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스: 05-2.app.py → app.py 로 복사 (Gunicorn 모듈명 호환)
COPY 05.Streamlit/05-2.server.py ./app.py

# PDF 및 캐시 파일 복사
COPY Data/ ./Data/

# vectorDB는 볼륨으로 마운트 (docker-compose.yml 참고)
RUN mkdir -p ./vectorDB

EXPOSE 8000

# Gunicorn + uvicorn worker 설정
# --preload: 마스터에서 앱 로드 후 포크 → VectorDB 초기화 1회만 실행
# --timeout 120: LLM 생성 시간 고려
# --workers: CPU * 2 + 1 공식 기준 (조정 가능)
CMD ["gunicorn", "app:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "180", \
     "--preload"]
