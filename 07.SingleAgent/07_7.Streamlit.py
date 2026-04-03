import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Single Agent Curriculum", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# 세션 유형 스타일 매핑
# ---------------------------------------------------------------------------

_SESSION_TYPE_STYLE = {
    "공통 이론":       ("#dbeafe", "#1e40af", "📖"),
    "공통 실습":       ("#dcfce7", "#166534", "🛠"),
    "그룹별 프로젝트": ("#fef3c7", "#92400e", "🧩"),
    "그룹별 심화 적용":("#fce7f3", "#9d174d", "🔬"),
}

_GROUP_COLORS = {
    "그룹 A": ("#dbeafe", "#1e40af"),
    "그룹 B": ("#dcfce7", "#166534"),
    "그룹 C": ("#fef3c7", "#92400e"),
}


def _type_badge(session_type: str) -> str:
    bg, tc, icon = _SESSION_TYPE_STYLE.get(session_type, ("#f1f5f9", "#334155", "📌"))
    return (
        f'<span style="background:{bg};color:{tc};padding:2px 8px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600">'
        f'{icon} {session_type}</span>'
    )


def _pill(text: str, bg: str = "#f1f5f9", tc: str = "#334155") -> str:
    return (
        f'<span style="background:{bg};color:{tc};padding:1px 8px;'
        f'border-radius:10px;font-size:0.8em;font-weight:500">{text}</span>'
    )

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


def call_chat(payload: dict) -> dict:
    r = requests.post(_url("/chat"), json=payload, headers=_headers(), timeout=600)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "token": None,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 기업 맞춤형 AI 교육 커리큘럼 설계를 도와드릴게요.\n"
                    "먼저 회사명과 교육 목표를 알려주세요."
                ),
            }
        ],
        "curriculum_plan": None,
        "curriculum_complete": True,
        "curriculum_validation": "",
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
# 로그인
# ---------------------------------------------------------------------------

def render_login():
    st.title("Single Agent Curriculum 로그인")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        if st.form_submit_button("로그인", use_container_width=True):
            try:
                st.session_state.token = call_login(username, password)
                st.rerun()
            except requests.HTTPError as e:
                st.error(f"로그인 실패: {e.response.text if e.response else e}")

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Single Agent")
        health = check_backend_health(st.session_state.get("token", ""))
        if health:
            st.success(f"백엔드 연결됨  ·  {health['chunks']}개 청크")
        else:
            st.error("백엔드 연결 실패")

        st.divider()
        if st.button("새 대화 시작", use_container_width=True):
            reset_conversation()
            st.rerun()
        if st.button("로그아웃", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# 채팅 메시지
# ---------------------------------------------------------------------------

def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# 세션 카드 렌더러
# ---------------------------------------------------------------------------

def _render_session_card(session: dict, expanded: bool = False):
    s_type   = session.get("session_type", "")
    title    = session.get("title", "")
    target   = session.get("target", "")
    hours    = session.get("duration_hours", 0)
    purpose  = session.get("purpose", "")
    goals    = session.get("goals", [])
    # contents가 없으면 activities로 폴백
    contents = session.get("contents") or session.get("activities", [])
    method   = session.get("method", "")
    effect   = session.get("expected_effect", "")

    bg, tc, _ = _SESSION_TYPE_STYLE.get(s_type, ("#f1f5f9", "#334155", "📌"))

    header_html = (
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'{_type_badge(s_type)}'
        f'{_pill(f"👤 {target}")}'
        f'{_pill(f"⏱ {hours}h", "#f0fdf4", "#166534")}'
        f'</div>'
        f'<div style="font-weight:600;font-size:1em;margin-top:4px">{title}</div>'
    )

    with st.expander(f"{'  ' * 0}{title}  ({hours}h)", expanded=expanded):
        st.markdown(header_html, unsafe_allow_html=True)

        if purpose:
            st.markdown(
                f'<div style="background:{bg};color:{tc};padding:8px 12px;'
                f'border-radius:6px;margin:8px 0;font-size:0.9em">'
                f'<strong>목적</strong>  {purpose}</div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            if goals:
                st.markdown("**🎯 학습목표**")
                for g in goals:
                    st.markdown(f"- {g}")
            if method:
                st.markdown(f"**🔧 진행 방식**  \n{method}")

        with col2:
            if contents:
                st.markdown("**📚 학습내용**")
                for c in contents:
                    st.markdown(f"- {c}")
            if effect:
                st.markdown(
                    f'<div style="background:#f0fdf4;border-left:3px solid #22c55e;'
                    f'padding:6px 10px;border-radius:0 4px 4px 0;margin-top:8px;font-size:0.88em">'
                    f'<strong>💡 기대효과</strong><br>{effect}</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# 커리큘럼 렌더러
# ---------------------------------------------------------------------------

def render_curriculum():
    plan = st.session_state.get("curriculum_plan")
    if not plan:
        return

    st.divider()

    # ── 0. 검증 상태 배너 ────────────────────────────────────────────
    is_complete = st.session_state.get("curriculum_complete", True)
    if not is_complete:
        validation = st.session_state.get("curriculum_validation", "")
        fail_lines = [l.strip() for l in validation.splitlines() if l.strip().startswith("-")]
        detail = "\n".join(fail_lines) if fail_lines else ""
        st.warning(
            "**임시 결과** — 구조 검증을 통과하지 못한 커리큘럼입니다. "
            "수정 요청을 입력하면 재생성합니다."
            + (f"\n\n검증 실패 사유:\n{detail}" if detail else ""),
            icon="⚠️",
        )

    # ── 1. 과정 개요 배너 ────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);'
        f'color:white;padding:16px 20px;border-radius:10px;margin-bottom:16px">'
        f'<div style="font-size:1.3em;font-weight:700">🎓 {plan.get("program_title","")}</div>'
        f'<div style="opacity:0.85;margin-top:4px">{plan.get("target_summary","")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 2. 그룹 구성 ─────────────────────────────────────────────────
    groups = plan.get("groups", [])
    if groups:
        total = sum(g.get("participant_count", 0) for g in groups)
        m_cols = st.columns(1 + len(groups))
        m_cols[0].metric("총 인원", f"{total}명")
        for i, g in enumerate(groups):
            m_cols[i + 1].metric(g["group_name"], f"{g['participant_count']}명",
                                  help=f"{g.get('target_types','')}  |  {g.get('focus_description','')}")

        with st.expander("👥 그룹별 설계 방향", expanded=False):
            gcols = st.columns(len(groups))
            for col, g in zip(gcols, groups):
                bg, tc = _GROUP_COLORS.get(g["group_name"], ("#f1f5f9", "#334155"))
                col.markdown(
                    f'<div style="background:{bg};border-left:3px solid {tc};'
                    f'padding:10px 12px;border-radius:0 6px 6px 0">'
                    f'<strong style="color:{tc}">{g["group_name"]}</strong>'
                    f'<br><span style="font-size:0.85em">{g.get("target_types","")}</span>'
                    f'<br><br>{g.get("focus_description","")}</div>',
                    unsafe_allow_html=True,
                )

    # ── 3. 일차별 커리큘럼 ───────────────────────────────────────────
    daily = plan.get("daily_schedules", [])
    if not daily:
        return

    st.markdown("### 📅 일차별 커리큘럼")
    day_tabs = st.tabs([f"Day {d['day']}  —  {d.get('theme','')}" for d in daily])

    for tab, day in zip(day_tabs, daily):
        with tab:
            common         = day.get("common_sessions", [])
            group_sessions = day.get("group_sessions", [])

            common_hours = sum(s.get("duration_hours", 0) for s in common)
            group_hours  = sum(
                s.get("duration_hours", 0)
                for gs in group_sessions[:1]
                for s in gs.get("sessions", [])
            )
            total_h = common_hours + group_hours

            # 일차 요약 헤더
            st.markdown(
                f'<div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap">'
                f'{_pill(f"총 {total_h}h", "#f1f5f9", "#334155")}'
                f'{_pill(f"공통 {common_hours}h", "#dbeafe", "#1e40af")}'
                f'{_pill(f"그룹 실습 {group_hours}h", "#fef3c7", "#92400e")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 공통 세션
            if common:
                st.markdown("#### 📖 공통 세션")
                for i, s in enumerate(common):
                    _render_session_card(s, expanded=(i == 0))

            # 그룹별 세션
            if group_sessions:
                st.markdown("#### 🧩 그룹별 병렬 세션")
                st.caption("아래 그룹은 동일 시간대에 각자의 세션을 병렬 진행합니다.")
                gcols = st.columns(len(group_sessions))
                for col, gs in zip(gcols, group_sessions):
                    g_name = gs.get("group_name", "")
                    bg, tc = _GROUP_COLORS.get(g_name, ("#f1f5f9", "#334155"))
                    g_hours = sum(s.get("duration_hours", 0) for s in gs.get("sessions", []))
                    with col:
                        st.markdown(
                            f'<div style="background:{bg};color:{tc};padding:6px 12px;'
                            f'border-radius:6px;font-weight:600;margin-bottom:8px">'
                            f'{g_name}  ·  {g_hours}h</div>',
                            unsafe_allow_html=True,
                        )
                        for s in gs.get("sessions", []):
                            _render_session_card(s, expanded=False)

    # ── 4. 예상 결과 & 참고 사항 ─────────────────────────────────────
    outcomes = plan.get("expected_outcomes", [])
    notes    = plan.get("notes", [])
    if outcomes or notes:
        st.markdown("### 📊 전체 기대효과 & 참고사항")
        r_col, n_col = st.columns(2)
        with r_col:
            st.markdown("**🎯 전체 기대효과**")
            for o in outcomes:
                st.success(o)
        with n_col:
            st.markdown("**📝 참고 사항**")
            for n in notes:
                st.warning(n)

# ---------------------------------------------------------------------------
# 메시지 처리
# ---------------------------------------------------------------------------

def handle_user_message(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    try:
        with st.spinner("에이전트가 처리 중입니다..."):
            result = call_chat({"messages": st.session_state.messages})
    except requests.HTTPError as e:
        detail = e.response.text if e.response else str(e)
        st.session_state.messages.append({"role": "assistant", "content": f"오류: {detail}"})
        return
    except requests.ConnectionError:
        st.session_state.messages.append({"role": "assistant", "content": "백엔드 서버에 연결할 수 없습니다."})
        return
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"예상하지 못한 오류: {e}"})
        return

    if result.get("curriculum"):
        complete = result.get("complete", False)
        validation_result = result.get("validation_result") or ""
        st.session_state.curriculum_plan = result["curriculum"]
        st.session_state.curriculum_complete = complete
        st.session_state.curriculum_validation = validation_result

        if complete:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "커리큘럼이 생성되었습니다! 아래에서 확인하세요. 수정이 필요하시면 말씀해 주세요.",
            })
        else:
            # 검증 실패: 검증 사유를 사용자에게 노출
            fail_lines = [l for l in validation_result.splitlines() if l.strip().startswith("-")]
            fail_summary = "\n".join(fail_lines) if fail_lines else "구조 검증에 실패했습니다."
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "커리큘럼이 생성되었으나 구조 검증을 통과하지 못했습니다. "
                    "아래에서 임시 결과를 확인하실 수 있으며, 수정 요청 시 재생성합니다.\n\n"
                    f"**검증 실패 사유:**\n{fail_summary}"
                ),
            })
    else:
        st.session_state.messages.append({"role": "assistant", "content": result["reply"]})

# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    init_session_state()

    if not st.session_state.get("token"):
        render_login()
        return

    render_sidebar()
    st.title("Single Agent 커리큘럼 설계")
    st.caption("사용자와 대화하며 요구사항을 수집하고 맞춤형 AI 교육 커리큘럼을 설계합니다.")

    render_messages()
    render_curriculum()

    user_text = st.chat_input("기업의 요구사항이나 수정 요청을 입력해 주세요.")
    if user_text:
        handle_user_message(user_text)
        st.rerun()


if __name__ == "__main__":
    main()
