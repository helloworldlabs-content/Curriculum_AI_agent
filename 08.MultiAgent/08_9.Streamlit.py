import json
import os

import pandas as pd
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
                "evaluating": "품질 평가",  "complete": "완료"}

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
    "previous_curriculum": None,
    "regen_count": 0,
    "max_regen": 3,
    "last_evaluator_feedback": "",
    "similarity_context": "",
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
# 파이프라인 상태
# ---------------------------------------------------------------------------

def render_pipeline_status():
    state       = st.session_state.get("orchestrator_state", {})
    current     = state.get("phase", "collecting")
    regen_count = state.get("regen_count", 0)
    max_regen   = state.get("max_regen", 3)
    current_idx = PHASE_ORDER.index(current)

    cols = st.columns(len(PHASE_ORDER))
    for i, (phase, col) in enumerate(zip(PHASE_ORDER, cols)):
        icon, label = PHASE_ICONS[phase], PHASE_LABELS[phase]
        if i < current_idx:
            col.success(f"{icon} {label}")
        elif i == current_idx:
            col.info(f"{icon} **{label}** ←")
        else:
            col.markdown(f"⬜ {label}")

    if regen_count > 0:
        st.caption(f"🔄 커리큘럼 재생성: {regen_count} / {max_regen}회")

# ---------------------------------------------------------------------------
# 평가 이력
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
            badge_cols = st.columns(4)
            for col, (key, name) in zip(badge_cols, EVAL_KEYS):
                ok = record.get(key, True)
                col.metric(name, "✅ 통과" if ok else "❌ 미통과")

            st.markdown(f"**평가 요약**\n{record.get('summary', '')}")

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
# 표 형식 커리큘럼 렌더러
# ---------------------------------------------------------------------------

def _sessions_to_df(common_sessions: list, group_sessions: list) -> pd.DataFrame:
    """공통 세션과 그룹 세션을 하나의 DataFrame으로 변환한다."""
    rows = []
    for s in common_sessions:
        rows.append({
            "구분":     "공통",
            "세션명":   s.get("title", ""),
            "시간(h)":  s.get("duration_hours", 0),
            "학습 목표": "\n".join(f"• {g}" for g in s.get("goals", [])),
            "활동 내용": "\n".join(f"• {a}" for a in s.get("activities", [])),
        })
    for gs in group_sessions:
        group_name = gs.get("group_name", "")
        for s in gs.get("sessions", []):
            rows.append({
                "구분":     group_name,
                "세션명":   s.get("title", ""),
                "시간(h)":  s.get("duration_hours", 0),
                "학습 목표": "\n".join(f"• {g}" for g in s.get("goals", [])),
                "활동 내용": "\n".join(f"• {a}" for a in s.get("activities", [])),
            })
    return pd.DataFrame(rows)


def render_curriculum_table(plan: dict) -> None:
    """커리큘럼을 표 형식으로 렌더링한다."""
    groups = plan.get("groups", [])
    daily  = plan.get("daily_schedules", [])

    # ── 1. 그룹 구성 표 ──────────────────────────────────────────────
    st.markdown("#### 👥 그룹 구성")
    if groups:
        total = sum(g.get("participant_count", 0) for g in groups)
        df_groups = pd.DataFrame([{
            "그룹":     g.get("group_name", ""),
            "대상 유형": g.get("target_types", ""),
            "인원":     f"{g.get('participant_count', 0)}명",
            "교육 방향": g.get("focus_description", ""),
        } for g in groups])
        st.dataframe(df_groups, hide_index=True, use_container_width=True)
        st.caption(f"총 {total}명")

    # ── 2. 일차별 세션 표 ────────────────────────────────────────────
    st.markdown("#### 📅 일차별 세션")

    if not daily:
        st.warning("일차별 스케줄 데이터가 없습니다.")
        return

    day_tabs = st.tabs([f"Day {d['day']}  —  {d.get('theme', '')}" for d in daily])

    for tab, day in zip(day_tabs, daily):
        with tab:
            common        = day.get("common_sessions", [])
            group_sessions = day.get("group_sessions", [])
            common_hours  = sum(s.get("duration_hours", 0) for s in common)
            group_hours   = sum(
                s.get("duration_hours", 0)
                for gs in group_sessions[:1]
                for s in gs.get("sessions", [])
            )
            st.caption(f"공통 {common_hours}h  +  그룹별 실습 {group_hours}h  =  총 {common_hours + group_hours}h")

            df = _sessions_to_df(common, group_sessions)
            if not df.empty:
                # 구분 컬럼 색상 강조를 위해 스타일 적용
                group_colors = {
                    "공통":   "#e8f5e9",
                    "그룹 A": "#e3f2fd",
                    "그룹 B": "#fff3e0",
                    "그룹 C": "#fce4ec",
                }

                def _row_color(row):
                    color = group_colors.get(row["구분"], "#ffffff")
                    return [f"background-color: {color}"] * len(row)

                styled = df.style.apply(_row_color, axis=1)
                st.dataframe(
                    styled,
                    hide_index=True,
                    use_container_width=True,
                    height=min(80 + len(df) * 80, 600),
                    column_config={
                        "구분":     st.column_config.TextColumn("구분",     width="small"),
                        "세션명":   st.column_config.TextColumn("세션명",   width="medium"),
                        "시간(h)":  st.column_config.NumberColumn("시간(h)", width="small", format="%.1f h"),
                        "학습 목표": st.column_config.TextColumn("학습 목표", width="large"),
                        "활동 내용": st.column_config.TextColumn("활동 내용", width="large"),
                    },
                )

    # ── 3. 예상 결과 & 참고 사항 표 ──────────────────────────────────
    outcomes = plan.get("expected_outcomes", [])
    notes    = plan.get("notes", [])
    if outcomes or notes:
        st.markdown("#### 📊 예상 결과 & 참고 사항")
        r_col, n_col = st.columns(2)
        with r_col:
            st.markdown("**🎯 예상 결과**")
            if outcomes:
                df_out = pd.DataFrame({"예상 결과": outcomes})
                st.dataframe(df_out, hide_index=True, use_container_width=True)
        with n_col:
            st.markdown("**📝 참고 사항**")
            if notes:
                df_notes = pd.DataFrame({"참고 사항": notes})
                st.dataframe(df_notes, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 커리큘럼 메인 렌더러
# ---------------------------------------------------------------------------

def render_curriculum() -> None:
    plan = st.session_state.get("curriculum_plan")
    if not plan:
        return

    st.divider()
    st.markdown(f"### 📋 {plan.get('program_title', '')}")
    st.caption(plan.get("target_summary", ""))

    # 헤더: 그룹 인원 메트릭 + 다운로드 버튼
    groups = plan.get("groups", [])
    total  = sum(g.get("participant_count", 0) for g in groups)
    m_cols = st.columns(1 + len(groups) + 1)
    m_cols[0].metric("총 인원", f"{total}명")
    for i, g in enumerate(groups):
        m_cols[i + 1].metric(g["group_name"], f"{g['participant_count']}명")

    json_str = json.dumps(plan, ensure_ascii=False, indent=2)
    m_cols[-1].download_button(
        label="📥 JSON 다운로드",
        data=json_str.encode("utf-8"),
        file_name=f"{plan.get('program_title', 'curriculum')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # 뷰 전환 탭
    tab_table, tab_json = st.tabs(["📊 표 형식", "📄 JSON"])

    with tab_table:
        render_curriculum_table(plan)

    with tab_json:
        st.caption("커리큘럼 전체 구조를 JSON으로 확인합니다.")
        st.json(plan, expanded=2)

# ---------------------------------------------------------------------------
# 메시지 처리
# ---------------------------------------------------------------------------

def handle_user_message(user_text: str) -> None:
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
# 로그인
# ---------------------------------------------------------------------------

def render_login() -> None:
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

def main() -> None:
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
