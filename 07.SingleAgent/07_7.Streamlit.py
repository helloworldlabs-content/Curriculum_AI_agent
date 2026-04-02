import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(page_title="Single Agent Curriculum", page_icon="AI", layout="wide")


def _url(path: str) -> str:
    return os.getenv("BACKEND_URL", "").rstrip("/") + path


def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}


@st.cache_data(ttl=30)
def check_backend_health(token: str) -> dict | None:
    try:
        response = requests.get(
            _url("/health"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return response.json() if response.ok else None
    except Exception:
        return None


def call_login(username: str, password: str) -> str:
    response = requests.post(
        _url("/auth/login"),
        json={"username": username, "password": password},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def call_chat(payload: dict) -> dict:
    response = requests.post(
        _url("/chat"),
        json=payload,
        headers=_headers(),
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def init_session_state():
    defaults = {
        "token": None,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요. 기업 맞춤형 AI 교육 커리큘럼 설계를 도와드릴게요. "
                    "먼저 회사명과 교육 목표를 알려주세요."
                ),
            }
        ],
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


def render_login():
    st.title("Single Agent 로그인")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인", use_container_width=True)
        if submitted:
            try:
                st.session_state.token = call_login(username, password)
                st.success("로그인에 성공했습니다.")
                st.rerun()
            except requests.HTTPError as error:
                detail = error.response.text if error.response is not None else str(error)
                st.error(f"로그인 실패: {detail}")


def render_sidebar():
    with st.sidebar:
        st.header("Single Agent")
        health = check_backend_health(st.session_state.get("token", ""))
        if health:
            st.success(f"백엔드 연결됨: {health['chunks']}개 청크")
        else:
            st.error("백엔드 연결 실패")

        if st.button("새 대화 시작", use_container_width=True):
            reset_conversation()
            st.rerun()

        if st.button("로그아웃", use_container_width=True):
            st.session_state.clear()
            st.rerun()



def render_messages():
    for message in st.session_state.messages:
        with st.chat_message("assistant" if message["role"] == "assistant" else "user"):
            st.write(message["content"])


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

    # 프로그램 제목 & 요약
    st.markdown(f"## {plan['program_title']}")
    st.caption(plan.get("target_summary", ""))

    # 그룹 인원 메트릭
    groups = plan.get("groups", [])
    total = sum(g.get("participant_count", 0) for g in groups)
    metric_cols = st.columns(1 + len(groups))
    metric_cols[0].metric("총 인원", f"{total}명")
    for i, g in enumerate(groups):
        metric_cols[i + 1].metric(g["group_name"], f"{g['participant_count']}명")

    daily_schedules = plan.get("daily_schedules", [])
    if not daily_schedules:
        st.warning("일차별 스케줄 데이터가 없습니다.")
        return

    # 일차별 탭
    tab_labels = [f"Day {d['day']}" for d in daily_schedules] + ["📊 결과 & 메모"]
    tabs = st.tabs(tab_labels)

    for tab, day in zip(tabs[:-1], daily_schedules):
        with tab:
            st.markdown(f"### {day['day']}일차 — {day.get('theme', '')}")

            common = day.get("common_sessions", [])
            group_sessions = day.get("group_sessions", [])

            common_hours = sum(s.get("duration_hours", 0) for s in common)
            group_hours = sum(
                s.get("duration_hours", 0)
                for gs in group_sessions[:1]
                for s in gs.get("sessions", [])
            )
            st.caption(
                f"공통 {common_hours}시간 + 그룹별 실습 {group_hours}시간 = 총 {common_hours + group_hours}시간"
            )

            # 공통 세션
            if common:
                st.markdown("#### 📖 공통 세션")
                for i, session in enumerate(common, 1):
                    _render_session_card(session, i)

            # 그룹별 실습
            if group_sessions:
                st.markdown("#### 🧩 그룹별 실습")
                gcols = st.columns(len(group_sessions))
                for col, gs in zip(gcols, group_sessions):
                    with col:
                        st.markdown(f"**{gs['group_name']}**")
                        for s in gs.get("sessions", []):
                            with st.expander(
                                f"{s['title']} · {s['duration_hours']}h", expanded=False
                            ):
                                st.markdown("**학습 목표**")
                                for goal in s.get("goals", []):
                                    st.markdown(f"- {goal}")
                                st.markdown("**활동 내용**")
                                for act in s.get("activities", []):
                                    st.markdown(f"- {act}")

    # 결과 & 메모 탭
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


def handle_user_message(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    payload = {
        "messages": st.session_state.messages,
    }

    try:
        with st.spinner("Single Agent가 다음 단계를 판단하고 있습니다..."):
            result = call_chat(payload)
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        st.session_state.messages.append({"role": "assistant", "content": f"처리 중 오류가 발생했습니다: {detail}"})
        return
    except requests.ConnectionError:
        st.session_state.messages.append({"role": "assistant", "content": "백엔드 서버에 연결할 수 없습니다."})
        return
    except Exception as error:
        st.session_state.messages.append({"role": "assistant", "content": f"예상하지 못한 오류가 발생했습니다: {error}"})
        return

    st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
    if result.get("curriculum"):
        st.session_state.curriculum_plan = result["curriculum"]


def main():
    init_session_state()
    if not st.session_state.get("token"):
        render_login()
        return

    render_sidebar()
    st.title("대화형 Single Agent")
    st.caption(
        "대화를 통해 요구사항을 수집하고, 6개 유형 인원 분포를 바탕으로 "
        "3개 그룹 실습/프로젝트가 포함된 커리큘럼을 설계합니다."
    )
    render_messages()
    render_curriculum()

    user_text = st.chat_input("기업의 요구사항이나 수정 요청을 입력해 주세요.")
    if user_text:
        handle_user_message(user_text)
        st.rerun()


if __name__ == "__main__":
    main()
