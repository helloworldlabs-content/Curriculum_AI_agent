import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Multi Agent Curriculum", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

PHASE_ORDER  = ["collecting", "generating", "evaluating", "complete"]
PHASE_ICONS  = {"collecting": "📋", "generating": "📝", "evaluating": "🔍", "complete": "✅"}
PHASE_LABELS = {"collecting": "정보 수집", "generating": "커리큘럼 생성",
                "evaluating": "품질 평가", "complete": "완료"}

EVAL_KEYS = [
    ("time_constraint_ok", "시간 적합성"),
    ("feasibility_ok",     "현실성"),
    ("condition_fit_ok",   "조건 적합성"),
    ("group_balance_ok",   "그룹 균형"),
]

# ---------------------------------------------------------------------------
# API 헬퍼
# ---------------------------------------------------------------------------

def _url(path: str) -> str:
    return os.getenv("BACKEND_URL", "").rstrip("/") + path


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('token', '')}"}


@st.cache_data(ttl=30)
def check_backend_health(token: str) -> dict | None:
    try:
        r = requests.get(_url("/health"), headers={"Authorization": f"Bearer {token}"}, timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def call_login(username: str, password: str) -> str:
    r = requests.post(_url("/auth/login"), json={"username": username, "password": password}, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


def call_chat(messages: list[dict], state: dict) -> dict:
    r = requests.post(
        _url("/chat"),
        json={"messages": messages, "state": state},
        headers=_headers(),
        timeout=600,
    )
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------

_DEFAULT_STATE = {
    "phase": "collecting",
    "collected_info": None,
    "curriculum": None,
    "regen_count": 0,
    "max_regen": 3,
    "last_evaluator_feedback": "",
    "evaluation_history": [],
    "agent_log": [],
}

_WELCOME = (
    "안녕하세요! 기업 맞춤형 AI 교육 커리큘럼 설계를 도와드릴게요.\n\n"
    "저는 4개의 전문 에이전트가 협력해 최적의 커리큘럼을 만들어 드립니다:\n"
    "**📋 정보 수집 → 📝 커리큘럼 생성 → 🔍 품질 평가 → ✅ 완료**\n\n"
    "먼저 회사명과 교육 목표를 알려주세요."
)


def init_session_state():
    defaults = {
        "token": None,
        "messages": [{"role": "assistant", "content": _WELCOME}],
        "orchestrator_state": dict(_DEFAULT_STATE),
        "curriculum_plan": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation():
    token = st.session_state.get("token")
    st.session_state.clear()
    init_session_state()
    st.session_state.token = token

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Multi Agent")

        health = check_backend_health(st.session_state.get("token", ""))
        if health:
            st.success(f"백엔드 연결됨: {health['chunks']}개 청크")
        else:
            st.error("백엔드 연결 실패")

        st.divider()

        # 에이전트 활동 로그
        logs = st.session_state.get("orchestrator_state", {}).get("agent_log", [])
        if logs:
            st.markdown("**에이전트 활동 로그**")
            for log in logs[-15:]:
                st.caption(log)
            st.divider()

        if st.button("새 대화 시작", use_container_width=True):
            reset_conversation()
            st.rerun()

        if st.button("로그아웃", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# 파이프라인 상태 표시
# ---------------------------------------------------------------------------

def render_pipeline_status():
    state = st.session_state.get("orchestrator_state", {})
    current_phase = state.get("phase", "collecting")
    regen_count   = state.get("regen_count", 0)
    max_regen     = state.get("max_regen", 3)
    current_idx   = PHASE_ORDER.index(current_phase)

    cols = st.columns(len(PHASE_ORDER))
    for i, (phase, col) in enumerate(zip(PHASE_ORDER, cols)):
        icon  = PHASE_ICONS[phase]
        label = PHASE_LABELS[phase]
        if i < current_idx:
            col.success(f"{icon} {label}")
        elif i == current_idx:
            col.info(f"{icon} **{label}** ←")
        else:
            col.markdown(f"⬜ {label}")

    if regen_count > 0:
        st.caption(f"🔄 커리큘럼 재생성: {regen_count} / {max_regen}회")

# ---------------------------------------------------------------------------
# 평가 이력 패널
# ---------------------------------------------------------------------------

def render_evaluation_history():
    state   = st.session_state.get("orchestrator_state", {})
    history = state.get("evaluation_history", [])
    if not history:
        return

    st.divider()
    st.markdown("### 🔍 평가 에이전트 이력")

    for record in history:
        attempt = record.get("attempt", "?")
        passed  = record.get("passed", False)
        label   = f"{'✅ 통과' if passed else '❌ 미통과'} — {attempt}차 평가"
        with st.expander(label, expanded=(not passed and record == history[-1])):
            # 4가지 기준 뱃지
            badge_cols = st.columns(4)
            for col, (key, name) in zip(badge_cols, EVAL_KEYS):
                ok = record.get(key, True)
                col.metric(name, "✅ 통과" if ok else "❌ 미통과")

            # 평가 요약
            st.markdown(f"**평가 요약**\n{record.get('summary', '')}")

            # 미통과 시 개선 요청사항
            feedback = record.get("feedback", "")
            if feedback and not passed:
                st.markdown("**커리큘럼 에이전트에게 전달된 개선 요청:**")
                st.info(feedback)

# ---------------------------------------------------------------------------
# 메시지 렌더링
# ---------------------------------------------------------------------------

def render_messages():
    for message in st.session_state.messages:
        role = "assistant" if message["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(message["content"])

# ---------------------------------------------------------------------------
# 커리큘럼 렌더링
# ---------------------------------------------------------------------------

def _render_session_card(session: dict, index: int):
    label = f"**세션 {index}** — {session['title']}  ·  {session['duration_hours']}시간"
    with st.expander(label, expanded=(index == 1)):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**학습 목표**")
            for goal in session.get("goals", []):
                st.markdown(f"- {goal}")
        with c2:
            st.markdown("**활동 내용**")
            for activity in session.get("activities", []):
                st.markdown(f"- {activity}")


def render_curriculum():
    plan = st.session_state.get("curriculum_plan")
    if not plan:
        return

    st.divider()
    st.markdown(f"## {plan['program_title']}")
    st.caption(plan.get("target_summary", ""))

    groups = plan.get("groups", [])
    total  = sum(g.get("participant_count", 0) for g in groups)
    metric_cols = st.columns(1 + len(groups))
    metric_cols[0].metric("총 인원", f"{total}명")
    for i, g in enumerate(groups):
        metric_cols[i + 1].metric(g["group_name"], f"{g['participant_count']}명")

    daily_schedules = plan.get("daily_schedules", [])
    if not daily_schedules:
        st.warning("일차별 스케줄 데이터가 없습니다.")
        return

    tab_labels = [f"Day {d['day']}" for d in daily_schedules] + ["📊 결과 & 메모"]
    tabs = st.tabs(tab_labels)

    for tab, day in zip(tabs[:-1], daily_schedules):
        with tab:
            st.markdown(f"### {day['day']}일차 — {day.get('theme', '')}")
            common         = day.get("common_sessions", [])
            group_sessions = day.get("group_sessions", [])

            common_hours = sum(s.get("duration_hours", 0) for s in common)
            group_hours  = sum(
                s.get("duration_hours", 0)
                for gs in group_sessions[:1]
                for s in gs.get("sessions", [])
            )
            total_hours = common_hours + group_hours
            st.caption(f"공통 {common_hours}시간 + 그룹별 실습 {group_hours}시간 = 총 {total_hours}시간")

            if common:
                st.markdown("#### 📖 공통 세션")
                for i, session in enumerate(common, 1):
                    _render_session_card(session, i)

            if group_sessions:
                st.markdown("#### 🧩 그룹별 실습")
                gcols = st.columns(len(group_sessions))
                for col, gs in zip(gcols, group_sessions):
                    with col:
                        st.markdown(f"**{gs['group_name']}**")
                        for s in gs.get("sessions", []):
                            with st.expander(f"{s['title']} · {s['duration_hours']}h", expanded=False):
                                st.markdown("**학습 목표**")
                                for goal in s.get("goals", []):
                                    st.markdown(f"- {goal}")
                                st.markdown("**활동 내용**")
                                for act in s.get("activities", []):
                                    st.markdown(f"- {act}")

    with tabs[-1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**예상 결과**")
            for outcome in plan.get("expected_outcomes", []):
                st.success(outcome)
        with c2:
            st.markdown("**참고 사항**")
            for note in plan.get("notes", []):
                st.warning(note)

# ---------------------------------------------------------------------------
# 메시지 처리
# ---------------------------------------------------------------------------

def handle_user_message(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})

    state = st.session_state.get("orchestrator_state", {})
    phase = state.get("phase", "collecting")
    spinner_label = {
        "collecting": "📋 정보 수집 에이전트가 처리 중...",
        "generating": "📝 커리큘럼 생성 에이전트가 검색 및 설계 중... (수 분 소요)",
        "evaluating": "🔍 평가 에이전트가 검토 중...",
        "complete":   "처리 중...",
    }.get(phase, "에이전트가 처리 중...")

    try:
        with st.spinner(spinner_label):
            result = call_chat(st.session_state.messages, state)
    except requests.HTTPError as error:
        detail = error.response.text if error.response else str(error)
        st.session_state.messages.append({"role": "assistant", "content": f"오류: {detail}"})
        return
    except requests.ConnectionError:
        st.session_state.messages.append({"role": "assistant", "content": "백엔드 서버에 연결할 수 없습니다."})
        return
    except Exception as error:
        st.session_state.messages.append({"role": "assistant", "content": f"예상하지 못한 오류: {error}"})
        return

    st.session_state.orchestrator_state = result.get("state", state)
    st.session_state.messages.append({"role": "assistant", "content": result["reply"]})

    if result.get("curriculum"):
        st.session_state.curriculum_plan = result["curriculum"]

# ---------------------------------------------------------------------------
# 로그인 화면
# ---------------------------------------------------------------------------

def render_login():
    st.title("Multi Agent Curriculum 로그인")
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("로그인", use_container_width=True):
            try:
                st.session_state.token = call_login(username, password)
                st.rerun()
            except requests.HTTPError as error:
                detail = error.response.text if error.response else str(error)
                st.error(f"로그인 실패: {detail}")

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    init_session_state()

    if not st.session_state.get("token"):
        render_login()
        return

    render_sidebar()

    st.title("Multi Agent 커리큘럼 설계")
    st.caption("4개의 전문 에이전트가 협력해 기업 맞춤형 AI 교육 커리큘럼을 설계합니다.")

    render_pipeline_status()
    st.divider()

    render_messages()
    render_evaluation_history()
    render_curriculum()

    user_text = st.chat_input("메시지를 입력해 주세요.")
    if user_text:
        handle_user_message(user_text)
        st.rerun()


if __name__ == "__main__":
    main()
