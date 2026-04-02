# 빌드 컨텍스트: 프로젝트 루트 (Curriculum_AI_agent/)
# 실행: docker build -f 07.SingleAgent/07_8.Dockerfile -t curriculum-agent .

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 소스 복사 (07_6.Main.py → main.py)
COPY 07.SingleAgent/07_2.AgentSchemas.py ./07_2.AgentSchemas.py
COPY 07.SingleAgent/07_3.AgentHelpers.py ./07_3.AgentHelpers.py
COPY 07.SingleAgent/07_4.SingleAgent.py  ./07_4.SingleAgent.py
COPY 07.SingleAgent/07_5.Auth.py         ./07_5.Auth.py
COPY 07.SingleAgent/07_6.Main.py         ./main.py

# 05.Advanced_RAG Indexing 모듈 (벡터스토어 초기화에 재사용)
COPY 05.Advanced_RAG/05_2.Schemas.py        ./05.Advanced_RAG/05_2.Schemas.py
COPY 05.Advanced_RAG/05_4.Indexing.py       ./05.Advanced_RAG/05_4.Indexing.py
COPY 05.Advanced_RAG/05_5.Retrieval.py      ./05.Advanced_RAG/05_5.Retrieval.py
COPY 06.Evaluation/06_2.EvalCommon.py       ./06.Evaluation/06_2.EvalCommon.py
COPY 06.Evaluation/06_3.RetrievalEval.py    ./06.Evaluation/06_3.RetrievalEval.py
COPY 06.Evaluation/06_4.FaithfulnessEval.py ./06.Evaluation/06_4.FaithfulnessEval.py
COPY 06.Evaluation/06_5.CoverageEval.py     ./06.Evaluation/06_5.CoverageEval.py
COPY 06.Evaluation/06_6.RuleEval.py         ./06.Evaluation/06_6.RuleEval.py

# 데이터 디렉터리 (vectorDB는 볼륨으로 마운트)
COPY Data/ ./Data/
RUN mkdir -p ./vectorDB

ENV APP_BASE_DIR=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "300"]
