import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="AI 커리큘럼 설계", page_icon="◼", layout="wide")

# --- CSS ---

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f2f2f2; }

    [data-testid="stSidebar"] { background-color: #111111 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    [data-testid="stSidebar"] hr { border-color: #333333 !important; }

    [data-testid="stSidebar"] .stButton > button {
        background-color: #ffffff !important; color: #111111 !important;
        border: none !important; border-radius: 6px !important; font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button div { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton > button:hover { background-color: #dddddd !important; }

    .main .block-container { padding-top: 24px !important; padding-bottom: 12px !important; max-width: 860px !important; }

    .page-header {
        display: flex; align-items: center; justify-content: space-between;
        background: #ffffff; border-radius: 12px; border: 1px solid #e5e5e5;
        padding: 16px 24px; margin-bottom: 16px;
    }
    .page-header-title { font-size: 22px; font-weight: 800; color: #111111; }
    .status-badge-on {
        display: inline-flex; align-items: center; gap: 6px;
        background: #111111; color: #ffffff; border-radius: 20px;
        padding: 5px 14px; font-size: 13px; font-weight: 700;
    }
    .status-dot { width: 8px; height: 8px; background: #aaaaaa; border-radius: 50%; display: inline-block; }

    .user-row { display: flex; justify-content: flex-end; margin: 10px 0; }
    .user-bubble {
        background: #111111; color: #ffffff; border-radius: 18px 18px 4px 18px;
        padding: 10px 16px; max-width: 68%; font-size: 14px; line-height: 1.65; word-break: break-word;
    }
    .assistant-row { display: flex; justify-content: flex-start; align-items: flex-start; gap: 10px; margin: 10px 0; }
    .assistant-avatar {
        width: 34px; height: 34px; background: #111111; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: #ffffff; font-size: 15px; flex-shrink: 0; margin-top: 2px;
    }
    .assistant-bubble {
        background: #f5f5f5; color: #111111; border-radius: 4px 18px 18px 18px;
        padding: 10px 16px; max-width: 68%; font-size: 14px; line-height: 1.65; word-break: break-word;
    }

    .info-card {
        background: #ffffff; border-radius: 12px; border: 1px solid #e5e5e5;
        padding: 18px 20px; margin-bottom: 12px;
    }

    .stButton > button {
        background: #111111 !important; color: #ffffff !important;
        border: none !important; border-radius: 6px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background: #333333 !important; }

    [data-testid="metric-container"] {
        border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px 14px; background: #ffffff;
    }
    [data-testid="stExpander"] { border: 1px solid #e5e5e5 !important; border-radius: 8px !important; margin-bottom: 6px; }
    [data-testid="stExpander"] summary { font-weight: 600; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)


# --- 백엔드 API ---

def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}

def _url(path: str) -> str:
    return os.getenv("BACKEND_URL", "").rstrip("/") + path



@st.cache_data(ttl=30)
def check_backend_health() -> dict | None:
    try:
        resp = requests.get(_url("/health"), headers=_headers(), timeout=5)
        return resp.json() if resp.ok else None
    except Exception:
        return None


def call_chat(messages: list[dict]) -> dict:
    resp = requests.post(_url("/chat"), json={"messages": messages}, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_generate(messages: list[dict], collected_info: dict) -> dict:
    resp = requests.post(
        _url("/generate"),
        json={"messages": messages, "collected_info": collected_info},
        headers=_headers(),
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# --- 세션 상태 ---

def init_session_state():
    defaults = {
        "messages":       [],
        "collected_info": None,
        "complete":       False,
        "generating":     False,
        "curriculum":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_conversation():
    st.session_state.clear()
    init_session_state()


# --- 렌더링 ---

def escape_html(text: str) -> str:
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>"))


def render_chat_bubbles():
    html = '<div style="padding-bottom:16px;">'
    for msg in st.session_state.messages:
        content_html = escape_html(msg["content"])
        if msg["role"] == "user":
            html += f'<div class="user-row"><div class="user-bubble">{content_html}</div></div>'
        else:
            html += (f'<div class="assistant-row">'
                     f'<div class="assistant-avatar">◼</div>'
                     f'<div class="assistant-bubble">{content_html}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_session_expander(session, index):
    label = f"**{index}회차** — {session['title']}  ·  {session['duration_hours']}시간"
    with st.expander(label, expanded=(index == 1)):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**학습 목표**")
            for g in session["goals"]:
                st.markdown(f"- {g}")
        with c2:
            st.markdown("**활동 내용**")
            for a in session["activities"]:
                st.markdown(f"- {a}")


def render_curriculum(curriculum):
    st.markdown("---")
    st.markdown(f"## {curriculum['program_title']}")
    st.caption(curriculum["target_summary"])

    groups = curriculum["group_sessions"]
    total  = sum(g["participant_count"] for g in groups)
    cols   = st.columns(4)
    with cols[0]:
        st.metric("총 인원", f"{total}명")
    for i, g in enumerate(groups):
        with cols[i + 1]:
            st.metric(g["group_name"], f"{g['participant_count']}명")

    tabs = st.tabs(["📖 공통 이론 수업", "◼ 그룹 A", "◻ 그룹 B", "▪ 그룹 C", "📊 결과 & 메모"])

    with tabs[0]:
        st.caption("모든 그룹이 동일하게 수강합니다.")
        for i, s in enumerate(curriculum["theory_sessions"], 1):
            render_session_expander(s, i)

    for idx, group in enumerate(groups):
        with tabs[idx + 1]:
            tags = " &nbsp; ".join(
                f'<span style="border:1.5px solid #111;border-radius:20px;padding:2px 10px;font-size:12px;font-weight:600;">{t}</span>'
                for t in group["target_types"].replace(" · ", "·").split("·")
            )
            st.markdown(
                f'<div class="info-card" style="margin-bottom:14px;">'
                f'<div style="font-size:16px;font-weight:700;margin-bottom:6px;">'
                f'{group["group_name"]} &nbsp; <span style="font-weight:400;color:#666;font-size:14px;">{group["participant_count"]}명</span></div>'
                f'<div style="margin-bottom:8px;">{tags}</div>'
                f'<div style="font-size:13px;color:#555;">실습 포커스: {group["focus_description"]}</div></div>',
                unsafe_allow_html=True,
            )
            for i, s in enumerate(group["sessions"], 1):
                render_session_expander(s, i)

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**예상 결과**")
            for o in curriculum["expected_outcomes"]:
                st.markdown(f'<div style="border-left:3px solid #111;padding:6px 12px;margin:4px 0;font-size:13px;background:#f9f9f9;">✓ {o}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**참고 사항**")
            for n in curriculum["notes"]:
                st.markdown(f'<div style="border-left:3px solid #aaa;padding:6px 12px;margin:4px 0;font-size:13px;background:#f9f9f9;">• {n}</div>', unsafe_allow_html=True)


# --- 사이드바 ---

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;letter-spacing:-0.5px;margin-bottom:4px;">◼ AI 커리큘럼</div>'
            '<div style="font-size:12px;color:#888;margin-bottom:20px;">AX Compass 기반 맞춤형 설계</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:#333;margin:0 0 16px 0;">', unsafe_allow_html=True)

        if st.button("새 대화 시작", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#777;letter-spacing:1px;margin-bottom:8px;">BACKEND</div>', unsafe_allow_html=True)
        health = check_backend_health()
        if health:
            st.markdown(f'<div style="color:#aaaaaa;font-size:13px;">✓ 연결됨 ({health["chunks"]}개 청크)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#ff6666;font-size:13px;">✗ 백엔드 연결 실패</div>', unsafe_allow_html=True)


# --- 인증 ---

def render_login():
    st.markdown("<h2 style='text-align:center;margin-top:80px;'>🔐 로그인</h2>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인", use_container_width=True)

    if submitted:
        try:
            resp = requests.post(
                _url("/auth/login"),
                json={"username": username, "password": password},
                timeout=10,
            )
            if resp.ok:
                st.session_state.token     = resp.json()["access_token"]
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        except Exception as e:
            st.error(f"백엔드 연결 실패: {e}")


# --- 메인 ---

def main():
    if not st.session_state.get("logged_in"):
        render_login()
        return

    # 사이드바 로그아웃
    with st.sidebar:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()


    apply_custom_css()
    init_session_state()
    render_sidebar()

    status = "완료" if st.session_state.curriculum else ("생성 중" if st.session_state.generating else "대화 중")
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-header-title">AI 커리큘럼 설계</div>'
        f'<div class="status-badge-on"><span class="status-dot"></span>{status}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 첫 인사 메시지
    if not st.session_state.messages:
        with st.spinner("연결 중..."):
            try:
                result = call_chat([])
                st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
            except Exception:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "안녕하세요! 기업 AI 교육 커리큘럼 설계를 도와드리겠습니다. 회사명과 교육 목표를 알려주세요.",
                })

    render_chat_bubbles()

    # 커리큘럼 생성 처리
    if st.session_state.generating:
        st.session_state.generating = False
        try:
            with st.spinner("커리큘럼을 생성하고 있습니다..."):
                curriculum = call_generate(st.session_state.messages, st.session_state.collected_info)
            st.session_state.curriculum = curriculum
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"✅ **{curriculum['program_title']}** 생성 완료! 아래에서 전체 커리큘럼을 확인하세요.",
            })
        except requests.HTTPError as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ 백엔드 오류 ({e.response.status_code})"})
        except requests.ConnectionError:
            st.session_state.messages.append({"role": "assistant", "content": "❌ 백엔드 서버에 연결할 수 없습니다."})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ 오류: {e}"})
        st.rerun()

    if st.session_state.curriculum:
        render_curriculum(st.session_state.curriculum)

    # 사용자 입력
    if not st.session_state.generating and not st.session_state.complete:
        if prompt := st.chat_input("메시지를 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner(""):
                    result = call_chat(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": result["reply"]})
                if result["complete"]:
                    st.session_state.collected_info = result["collected_info"]
                    st.session_state.complete       = True
                    st.session_state.generating     = True
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ 오류: {e}"})
            st.rerun()


main()
