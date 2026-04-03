<h1 align="center">「2026년 서울시 민간단체 협력형 매력일자리 사업」AI 개발자 직무부트캠프</h1>

---

## 실습 결과 폴더 안내

각 챕터의 실습 결과물과 실습 내용은 다음과 같습니다.

| 챕터 | 폴더명 | 실습 내용 | 바로가기 |
|:---:|:---:|:---:|:---:|
| Chapter.2 | `00.tetris` | 간단한 프롬프트로 테트리스 만들기 | [00.tetris](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/00.tetris) |
| Chapter.2 | `01.tetris` | 설계된 프롬프트로 테트리스 만들기 | [01.tetris](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/01.tetris) |
| Chapter.2 | `02.terminal_tetris` | 터미널 콘솔 스타일 테트리스 만들기 | [02.terminal_tetris](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/02.terminal_tetris) |
| Chapter.2 | `03.ax_curriculum_chatbot` | 기업교육 커리큘럼 생성 챗봇 만들기 | [03.ax_curriculum_chatbot](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/03.ax_curriculum_chatbot) |
| Chapter.3 | `04.RAG` | LangChain을 활용하여 챗봇에 RAG 적용하기 | [04.RAG](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/04.RAG) |
| Chapter.4 | `05.Advanced_RAG` | Streamlit 화면 구성 및 Docker 배포 실습 | [05.Advanced_RAG](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/05.Advanced_RAG) |
| Chapter.5 | `05.Advanced_RAG` | 인덱싱 파이프라인 고도화 및 Contextual Retrieval | [05.Advanced_RAG](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/05.Advanced_RAG) |
| Chapter.5 | `05.Advanced_RAG` | Hybrid Search 및 Reranking | [05.Advanced_RAG](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/05.Advanced_RAG) |
| Chapter.5 | `06.Evaluation` | RAG 성능 평가 | [06.Evaluation](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/06.Evaluation) |
| Chapter.6 | `07.SingleAgent` | LangGraph를 활용한 AI Agent 만들기 | [07.SingleAgent](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/07.SingleAgent) |
| Chapter.6 | `08.MultiAgent` | 오케스트레이터를 활용한 Multi Agent 만들기 | [08.MultiAgent](https://github.com/helloworldlabs-content/Curriculum_AI_agent/tree/main/08.MultiAgent) |


## 실행 방법

이 저장소는 루트 기준으로 `.venv` 가상환경을 만든 뒤, 공통 의존성을 설치하고 각 폴더별 예제를 실행하는 방식으로 사용하는 것을 권장합니다.

### 1. 가상환경 생성 및 활성화

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. 공통 패키지 설치

```powershell
pip install -r requirements.txt
```

### 3. 환경 변수 파일 준비

OpenAI 기반 예제는 루트에 `.env` 파일이 필요합니다.

```env
OPENAI_API_KEY=sk-...
BACKEND_API_KEY=...
BACKEND_URL=...

# 인증 (streamlit-authenticator)
# AUTH_PASSWORD_HASH: bcrypt 해시값 (python -c "import bcrypt; print(bcrypt.hashpw('비밀번호'.encode(), bcrypt.gensalt()).decode())")

AUTH_USERNAME=admin
AUTH_PASSWORD_HASH=...
TAVILY_API_KEY=tvly-...
```

### 4. 폴더별 실행 명령어

아래 명령은 저장소 루트에서 실행하는 기준입니다.

| 폴더 | 설명 | 실행 명령어 |
|---|---|---|
| `00.tetris` | 간단한 테트리스 예제 | `python 00.tetris/tetris.py` |
| `01.tetris` | pygame 기반 테트리스 | `python 01.tetris/main.py` |
| `02.terminal_tetris` | 터미널 테트리스 | `python 02.terminal_tetris/tetris.py` |
| `03.ax_curriculum_chatbot` | 콘솔형 커리큘럼 챗봇 | `python 03.ax_curriculum_chatbot/ax_curriculum_chatbot.py` |
| `04.RAG` | 단일 파일 RAG 예제 | `python 04.RAG/04.RAG.py` |
| `05.Advanced_RAG` 백엔드 | FastAPI 서버 | `python 05.Advanced_RAG/05_6.Main.py` |
| `05.Advanced_RAG` 프런트 | Streamlit UI | `streamlit run 05.Advanced_RAG/05_1.Streamlit.py` |
| `06.Evaluation` | RAG 평가 실행 | `python 06.Evaluation/06_7.RunEval.py --testset 06.Evaluation/06_8.TestsetTemplate.json` |
| `07.SingleAgent` 백엔드 | Single Agent FastAPI 서버 | `python 07.SingleAgent/07_6.Main.py` |
| `07.SingleAgent` 프런트 | Single Agent Streamlit UI | `streamlit run 07.SingleAgent/07_7.Streamlit.py` |
| `08.MultiAgent` 백엔드 | Multi Agent FastAPI 서버 | `python 08.MultiAgent/08_8.Main.py` |
| `08.MultiAgent` 프런트 | Multi Agent Streamlit UI | `streamlit run 08.MultiAgent/08_9.Streamlit.py` |


## 교안 목차

### Chapter.1

*1. 오리엔테이션*

*2. AI 시대의 이해와 개발 직무의 변화*

*3. 나의 성장 목표 설정*

*4. AI 서비스 개발자에게 필요한 기본 지식*

*5. 문제정의 워크샵*

*6. 사용자 시나리오 설계*

### Chapter.2

*7. AI 개발 도구와 Vibe Coding 이해하기*

*8. 개발자 관점의 바이브 코딩*

*9. 프롬프트 엔지니어링의 이해와 기법*

*10. 기업교육 커리큘럼 생성봇 만들기*

### Chapter.3

*11. 단순 LLM 챗봇의 한계와 RAG 적용*

### Chapter.4

*12. Streamlit으로 서비스 화면 구성과 사용자 인증*

*13. Docker와 배포 도구를 활용한 서비스 배포 이해*

### Chapter.5

*14. 검색 품질을 높이는 인덱싱 전략과 Contextual Retrieval*

*15. 복잡한 질문에 대응하는 Retrieval 고도화: Hybrid Search와 Reranking*

*16. 생성 결과를 검증하고 개선하는 방법*

### Chapter.6

*17. Agent로 판단하고 실행하는 AI Workflow 적용*

*18. 복잡한 문제 해결을 위한 Multi Agent 적용*
