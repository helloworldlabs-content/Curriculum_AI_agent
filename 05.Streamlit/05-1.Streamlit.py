import re
import json
import os
import base64
from textwrap import dedent

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

st.set_page_config(page_title="AI 커리큘럼 설계", page_icon="◼", layout="wide")

# --- 경로 및 상수 ---

BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_PATH        = os.path.join(BASE_DIR, "Data", "AXCompass.pdf")
PDF_CACHE_PATH  = os.path.join(BASE_DIR, "Data", "ax_compass_full.txt")
VECTOR_DB_PATH  = os.path.join(BASE_DIR, "vectorDB")
COLLECTION_NAME = "ax_compass_types"

STAGE_GREETING     = "greeting"
STAGE_REQUIREMENTS = "requirements"
STAGE_TYPE_COUNTS  = "type_counts"
STAGE_CONFIRM      = "confirm"
STAGE_DONE         = "done"

REQUIREMENT_FIELDS = [
    ("company_name", "회사명 또는 팀 이름을 알려주세요."),
    ("goal",         "교육 목표가 무엇인지 알려주세요."),
    ("audience",     "교육 대상자는 누구인가요?"),
    ("level",        "현재 AI 활용 수준을 알려주세요.\n> 예시: 입문 / 초급 / 중급 / 고급"),
    ("duration",     "교육 기간 또는 총 시간은 얼마인가요?\n> 예시: 2일 / 16시간 / 4주"),
    ("topic",        "원하는 핵심 주제는 무엇인가요?"),
    ("constraints",  "꼭 반영해야 할 조건 또는 제한사항이 있나요?\n> 없으면 '없음'을 입력해주세요."),
]

TYPE_FIELDS = ["균형형", "이해형", "과신형", "실행형", "판단형", "조심형"]

TYPE_INFO = {
    "균형형": {"group": "A", "english": "BALANCED"},
    "이해형": {"group": "A", "english": "LEARNER"},
    "과신형": {"group": "B", "english": "OVERCONFIDENT"},
    "실행형": {"group": "B", "english": "DOER"},
    "판단형": {"group": "C", "english": "ANALYST"},
    "조심형": {"group": "C", "english": "CAUTIOUS"},
}

TYPE_MARKERS = {
    "균형형": "## 1) 균형형",
    "실행형": "## 2) 실행형",
    "판단형": "## 3) 판단형",
    "이해형": "## 4) 이해형",
    "과신형": "## 5) 과신형",
    "조심형": "## 6) 조심형",
}

_SESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "title":      {"type": "string"},
        "goals":      {"type": "array", "items": {"type": "string"}},
        "activities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "goals", "activities"],
    "additionalProperties": False,
}

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

    .step-item { display: flex; align-items: center; gap: 10px; padding: 7px 0; font-size: 13px; }
    .step-dot-done    { color: #888888; }
    .step-dot-current { color: #ffffff; font-weight: 700; }
    .step-dot-todo    { color: #444444; }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; flex-shrink: 0;
    }
    .step-num-done    { background: #444444; color: #888888; }
    .step-num-current { background: #ffffff; color: #111111; }
    .step-num-todo    { background: #333333; color: #555555; }

    button[data-baseweb="tab"] { font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111111 !important; border-bottom: 2px solid #111111 !important;
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


# --- 환경 설정 ---

def load_env_file():
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# --- RAG ---

def load_pdf_content():
    if os.path.exists(PDF_CACHE_PATH):
        with open(PDF_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF 파일 없음: {PDF_PATH}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(PDF_PATH, "rb") as f:
        pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": [
            {"type": "input_file", "filename": "AXCompass.pdf",
             "file_data": f"data:application/pdf;base64,{pdf_data}"},
            {"type": "input_text", "text":
             "이 PDF의 모든 내용을 한국어 마크다운으로 추출해줘. "
             "각 유형의 강점, 보완 방향, 대표 태그를 포함하고 "
             "각 유형 섹션은 '## 번호) 유형명' 형식으로 시작해줘."},
        ]}]
    )
    content = response.output_text
    os.makedirs(os.path.dirname(PDF_CACHE_PATH), exist_ok=True)
    with open(PDF_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def extract_type_chunks(pdf_content):
    chunks = []
    for type_name, marker in TYPE_MARKERS.items():
        start = pdf_content.find(marker)
        if start == -1:
            continue
        end = min(
            (pdf_content.find(om, start + len(marker))
             for om in TYPE_MARKERS.values() if om != marker
             if pdf_content.find(om, start + len(marker)) != -1),
            default=len(pdf_content),
        )
        info = TYPE_INFO[type_name]
        chunks.append({
            "id": type_name,
            "text": pdf_content[start:end].strip(),
            "metadata": {"type_name": type_name, "group": info["group"], "english": info["english"]},
        })
    return chunks


@st.cache_resource
def get_collection():
    load_env_file()
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        chunks = extract_type_chunks(load_pdf_content())
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
    return collection


def retrieve_type_context(collection, type_names):
    query = f"{', '.join(type_names)} 유형의 AI 활용 특성, 강점, 보완 방향, 교육적 접근 방법"
    results = collection.query(
        query_texts=[query], n_results=len(type_names),
        where={"type_name": {"$in": list(type_names)}},
        include=["documents"],
    )
    return "\n\n".join(results["documents"][0])


def generate_curriculum(requirements, groups, collection):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ga, gb, gc = groups["group_a"], groups["group_b"], groups["group_c"]
    ctxs = {
        "a": retrieve_type_context(collection, ga["types"]),
        "b": retrieve_type_context(collection, gb["types"]),
        "c": retrieve_type_context(collection, gc["types"]),
    }

    system_prompt = dedent("""
        당신은 기업 교육용 AI 커리큘럼 설계 전문가다.
        AX Compass 진단 결과와 유형별 특성을 바탕으로 맞춤형 교육 커리큘럼을 설계하라.
        - theory_sessions: 공통 이론 수업 4~6개
        - group_sessions: 3개 그룹별 실습 각 2~3개
        JSON만 반환. 마크다운 코드블록 없음.
    """).strip()

    req = requirements
    user_prompt = dedent(f"""
        [기업 요구사항]
        회사/팀: {req['company_name']} | 목표: {req['goal']}
        대상자: {req['audience']} | 수준: {req['level']}
        기간: {req['duration']} | 주제: {req['topic']}
        제한사항: {req['constraints']}

        [그룹 구성]
        - {ga['name']} ({' · '.join(ga['types'])}): {ga['count']}명
        - {gb['name']} ({' · '.join(gb['types'])}): {gb['count']}명
        - {gc['name']} ({' · '.join(gc['types'])}): {gc['count']}명

        [AX Compass 유형 특성]
        그룹 A: {ctxs['a']}
        그룹 B: {ctxs['b']}
        그룹 C: {ctxs['c']}
    """).strip()

    schema = {
        "type": "json_schema", "name": "curriculum_plan_rag", "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "program_title":    {"type": "string"},
                "target_summary":   {"type": "string"},
                "theory_sessions":  {"type": "array", "items": _SESSION_SCHEMA},
                "group_sessions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "group_name":        {"type": "string"},
                        "target_types":      {"type": "string"},
                        "participant_count": {"type": "integer"},
                        "focus_description": {"type": "string"},
                        "sessions":          {"type": "array", "items": _SESSION_SCHEMA},
                    },
                    "required": ["group_name", "target_types", "participant_count", "focus_description", "sessions"],
                    "additionalProperties": False,
                }},
                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                "notes":             {"type": "array", "items": {"type": "string"}},
            },
            "required": ["program_title", "target_summary", "theory_sessions",
                         "group_sessions", "expected_outcomes", "notes"],
            "additionalProperties": False,
        },
    }

    response = client.responses.create(
        model="gpt-4.1-mini", text={"format": schema},
        input=[{"role": "developer", "content": system_prompt},
               {"role": "user", "content": user_prompt}],
    )
    return json.loads(response.output_text.strip())


# --- 세션 상태 ---

def init_session_state():
    defaults = {
        "messages": [], "stage": STAGE_GREETING,
        "req_index": 0, "type_index": 0,
        "requirements": {}, "type_counts": {},
        "groups": {}, "curriculum": None, "generating": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_conversation():
    st.session_state.clear()
    init_session_state()


# --- 대화 로직 ---

def add_msg(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def handle_user_input(user_input):
    add_msg("user", user_input)
    stage = st.session_state.stage

    if stage == STAGE_GREETING:
        st.session_state.stage = STAGE_REQUIREMENTS
        st.session_state.req_index = 0
        _, q = REQUIREMENT_FIELDS[0]
        add_msg("assistant", f"시작할게요! 기업 요구사항을 순서대로 입력해주세요.\n\n**(1/7)** {q}")

    elif stage == STAGE_REQUIREMENTS:
        idx = st.session_state.req_index
        field, _ = REQUIREMENT_FIELDS[idx]
        st.session_state.requirements[field] = user_input
        st.session_state.req_index += 1
        nxt = st.session_state.req_index
        if nxt < len(REQUIREMENT_FIELDS):
            _, nq = REQUIREMENT_FIELDS[nxt]
            add_msg("assistant", f"**({nxt + 1}/7)** {nq}")
        else:
            st.session_state.stage = STAGE_TYPE_COUNTS
            st.session_state.type_index = 0
            add_msg("assistant",
                "요구사항 입력 완료! ✓\n\n"
                "이제 **AX Compass 진단 결과**를 입력해주세요.\n"
                f"**(1/6)** **{TYPE_FIELDS[0]}** 인원수를 입력해주세요.")

    elif stage == STAGE_TYPE_COUNTS:
        digits = re.sub(r"[^0-9]", "", user_input)
        if not digits:
            add_msg("assistant", "숫자를 입력해주세요. 예: `5`")
            return
        idx = st.session_state.type_index
        st.session_state.type_counts[TYPE_FIELDS[idx]] = int(digits)
        st.session_state.type_index += 1
        nxt = st.session_state.type_index
        if nxt < len(TYPE_FIELDS):
            add_msg("assistant", f"**({nxt + 1}/6)** **{TYPE_FIELDS[nxt]}** 인원수를 입력해주세요.")
        else:
            tc = st.session_state.type_counts
            groups = {
                "group_a": {"name": "그룹 A", "types": ["균형형", "이해형"], "count": tc["균형형"] + tc["이해형"]},
                "group_b": {"name": "그룹 B", "types": ["과신형", "실행형"], "count": tc["과신형"] + tc["실행형"]},
                "group_c": {"name": "그룹 C", "types": ["판단형", "조심형"], "count": tc["판단형"] + tc["조심형"]},
            }
            st.session_state.groups = groups
            total = sum(tc.values())
            req = st.session_state.requirements
            add_msg("assistant",
                "모든 정보 입력 완료! 아래 내용으로 커리큘럼을 생성할까요?\n\n"
                f"**회사/팀** {req['company_name']}  |  **기간** {req['duration']}  |  **수준** {req['level']}\n\n"
                f"**그룹 A** (균형형·이해형) {groups['group_a']['count']}명  "
                f"**그룹 B** (과신형·실행형) {groups['group_b']['count']}명  "
                f"**그룹 C** (판단형·조심형) {groups['group_c']['count']}명  "
                f"*(총 {total}명)*\n\n"
                "**'생성'** 을 입력하면 커리큘럼 생성을 시작합니다.")
            st.session_state.stage = STAGE_CONFIRM

    elif stage == STAGE_CONFIRM:
        if any(k in user_input.lower() for k in ["생성", "네", "yes", "y", "확인", "ok", "ㅇ", "응"]):
            add_msg("assistant", "커리큘럼을 생성하고 있습니다. 잠시만 기다려주세요...")
            st.session_state.stage = STAGE_DONE
            st.session_state.generating = True
        else:
            add_msg("assistant",
                "처음부터 다시 입력하려면 사이드바의 **새 대화 시작** 버튼을 눌러주세요.\n\n"
                "계속 진행하려면 **'생성'** 을 입력해주세요.")


# --- 렌더링 ---

def md_to_html(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.*?)`', r'<code style="background:#e8e8e8;padding:1px 4px;border-radius:3px;">\1</code>', text)
    text = re.sub(r'^> (.*)', r'<span style="border-left:3px solid #999;padding-left:8px;color:#666;">\1</span>', text, flags=re.MULTILINE)
    text = text.replace("\n\n", "</p><p style='margin:6px 0;'>").replace("\n", "<br>")
    return f"<p style='margin:0;'>{text}</p>"


def render_chat_bubbles():
    html = '<div style="padding-bottom:16px;">'
    for msg in st.session_state.messages:
        content_html = md_to_html(msg["content"])
        if msg["role"] == "user":
            html += f'<div class="user-row"><div class="user-bubble">{content_html}</div></div>'
        else:
            html += (f'<div class="assistant-row">'
                     f'<div class="assistant-avatar">◼</div>'
                     f'<div class="assistant-bubble">{content_html}</div></div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_session_expander(session, index):
    with st.expander(f"**{index}회차** — {session['title']}", expanded=(index == 1)):
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
    total = sum(g["participant_count"] for g in groups)
    cols = st.columns(4)
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
    stage_order  = [STAGE_GREETING, STAGE_REQUIREMENTS, STAGE_TYPE_COUNTS, STAGE_CONFIRM, STAGE_DONE]
    stage_labels = [
        (STAGE_REQUIREMENTS, "기업 요구사항"),
        (STAGE_TYPE_COUNTS,  "유형별 인원수"),
        (STAGE_CONFIRM,      "내용 확인"),
        (STAGE_DONE,         "커리큘럼 생성"),
    ]
    current = st.session_state.stage
    cur_idx = stage_order.index(current) if current in stage_order else 0

    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;letter-spacing:-0.5px;margin-bottom:4px;">◼ AI 커리큘럼</div>'
            '<div style="font-size:12px;color:#888;margin-bottom:20px;">AX Compass 기반 맞춤형 설계</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:#333;margin:0 0 16px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#777;letter-spacing:1px;margin-bottom:10px;">PROGRESS</div>', unsafe_allow_html=True)

        steps_html = ""
        for i, (s_stage, s_label) in enumerate(stage_labels, 1):
            s_idx = stage_order.index(s_stage)
            if s_idx < cur_idx:
                cls = "done"
            elif s_idx == cur_idx:
                cls = "current"
            else:
                cls = "todo"
            steps_html += (f'<div class="step-item step-dot-{cls}">'
                           f'<div class="step-num step-num-{cls}">{i}</div>'
                           f'<span>{s_label}</span></div>')
        st.markdown(steps_html, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
        if st.button("새 대화 시작", use_container_width=True):
            reset_conversation()
            st.rerun()

        st.markdown('<hr style="border-color:#333;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#777;letter-spacing:1px;margin-bottom:8px;">VECTOR DB</div>', unsafe_allow_html=True)
        try:
            col = get_collection()
            st.markdown(f'<div style="color:#aaaaaa;font-size:13px;">✓ {col.count()}개 유형 로드됨</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div style="color:#ff6666;font-size:13px;">✗ {e}</div>', unsafe_allow_html=True)


# --- 메인 ---

def main():
    load_env_file()
    apply_custom_css()
    init_session_state()
    render_sidebar()

    stage = st.session_state.stage
    status_text = {
        STAGE_GREETING: "대기 중", STAGE_REQUIREMENTS: "요구사항 수집 중",
        STAGE_TYPE_COUNTS: "인원수 입력 중", STAGE_CONFIRM: "확인 중", STAGE_DONE: "완료",
    }.get(stage, "")
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-header-title">AI 커리큘럼 설계</div>'
        f'<div class="status-badge-on"><span class="status-dot"></span>{status_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        add_msg("assistant",
            "안녕하세요! **기업 교육용 AI 커리큘럼 설계** 챗봇입니다.\n\n"
            "AX Compass 진단 결과를 바탕으로 3개 그룹 맞춤형 커리큘럼 초안을 생성합니다.\n\n"
            "준비되셨으면 **'시작'** 을 입력해주세요!")

    render_chat_bubbles()

    if st.session_state.generating:
        st.session_state.generating = False
        try:
            collection = get_collection()
            with st.spinner("벡터 DB 검색 및 커리큘럼 생성 중..."):
                curriculum = generate_curriculum(
                    st.session_state.requirements,
                    st.session_state.groups,
                    collection,
                )
            st.session_state.curriculum = curriculum
            add_msg("assistant", f"✅ **{curriculum['program_title']}** 생성 완료!\n\n아래에서 전체 커리큘럼을 확인하세요.")
        except Exception as e:
            add_msg("assistant", f"❌ 오류: {e}")
        st.rerun()

    if st.session_state.curriculum:
        render_curriculum(st.session_state.curriculum)

    if prompt := st.chat_input("메시지를 입력하세요..."):
        handle_user_input(prompt)
        st.rerun()


main()
