import importlib.util
import logging
import os
import sys

from langchain_chroma import Chroma

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

logger = logging.getLogger("multi_agent.orchestrator")

# 평가 기준 레이블
_EVAL_LABELS = {
    "time_constraint_ok": "시간 적합성",
    "feasibility_ok":     "현실성",
    "condition_fit_ok":   "조건 적합성",
    "group_balance_ok":   "그룹 균형",
}


SIMILARITY_THRESHOLD = 0.5  # 세션 제목 Jaccard 유사도가 이 값 이상이면 "유사" 판정


def _calculate_similarity(curr1: dict | None, curr2: dict | None) -> float:
    """두 커리큘럼의 세션 제목 집합으로 Jaccard 유사도를 계산한다 (0.0 ~ 1.0)."""
    if not curr1 or not curr2:
        return 0.0

    # 프로그램 제목이 완전히 같으면 즉시 1.0 반환
    if curr1.get("program_title") == curr2.get("program_title"):
        return 1.0

    def _titles(c: dict) -> set:
        result = set()
        for day in c.get("daily_schedules", []):
            for s in day.get("common_sessions", []):
                t = s.get("title", "").strip()
                if t:
                    result.add(t)
            for gs in day.get("group_sessions", []):
                for s in gs.get("sessions", []):
                    t = s.get("title", "").strip()
                    if t:
                        result.add(t)
        return result

    t1, t2 = _titles(curr1), _titles(curr2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _load(module_name: str, filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_agents():
    info       = _load("info_collector_08",   "08_4.InfoCollectorAgent.py")
    curriculum = _load("curriculum_agent_08", "08_5.CurriculumAgent.py")
    evaluator  = _load("evaluator_08",        "08_6.EvaluatorAgent.py")
    return info, curriculum, evaluator


def _get_schemas():
    return _load("schemas_08", "08_2.AgentSchemas.py")


def _eval_summary_lines(evaluation: dict) -> str:
    """평가 결과 dict를 사람이 읽기 좋은 여러 줄 문자열로 변환한다."""
    lines = []
    for key, label in _EVAL_LABELS.items():
        ok = evaluation.get(key, True)
        lines.append(f"  {'✅' if ok else '❌'} {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 오케스트레이터
# ---------------------------------------------------------------------------

def run_orchestrator(
    messages: list[dict],
    state_dict: dict,
    vectorstore: Chroma,
) -> dict:
    """
    에이전트 간 상태 기계를 실행한다.

    상태 전이:
        collecting  → generating  : InfoCollector가 정보 수집 완료
        generating  → evaluating  : CurriculumAgent가 커리큘럼 생성 완료
        evaluating  → generating  : EvaluatorAgent가 미통과 판정 (regen_count < max_regen)
        evaluating  → complete    : EvaluatorAgent가 통과 판정

    한 번의 HTTP 요청 안에서 사용자 입력 없이 전이 가능한 단계는
    연속으로 실행해 RTT를 줄인다.
    """
    schemas = _get_schemas()
    OrchestratorState = schemas.OrchestratorState
    state = OrchestratorState.model_validate(state_dict)

    info_mod, curriculum_mod, evaluator_mod = _get_agents()

    reply = ""
    active_agent = state.phase
    final_curriculum = state.curriculum

    # -----------------------------------------------------------------------
    # 단계 1: 정보 수집 (collecting)
    # -----------------------------------------------------------------------
    if state.phase == "collecting":
        active_agent = "info_collector"
        logger.info("[orchestrator] phase=collecting")

        result = info_mod.run_info_collector(
            messages,
            existing_info=state.collected_info,
            similarity_context=state.similarity_context,
        )
        reply = result["reply"]
        state.agent_log.append(f"[정보 수집 에이전트] {reply[:80]}")

        if result["complete"]:
            state.collected_info = result["collected_info"]
            state.phase = "generating"
            # 유사도로 인해 재수집한 경우 재생성 카운터·피드백·유사도 컨텍스트를 초기화
            state.regen_count = 0
            state.last_evaluator_feedback = ""
            state.similarity_context = ""
            logger.info("[orchestrator] info collected → transitioning to generating")

            # 정보 수집 완료 메시지 + 전환 안내를 앞에 붙임
            transition_notice = (
                "\n\n---\n"
                "**📋 정보 수집 에이전트 → 📝 커리큘럼 생성 에이전트**\n"
                "필요한 정보가 모두 수집되었습니다. 지금부터 커리큘럼을 생성합니다.\n"
                "RAG 검색(AX Compass, 커리큘럼 예시)과 웹 검색을 수행한 뒤 커리큘럼을 설계합니다...\n"
                "---"
            )
            state, chain_reply, final_curriculum, active_agent = _run_generating_chain(
                state, messages, vectorstore, curriculum_mod, evaluator_mod,
            )
            reply = reply + transition_notice + "\n\n" + chain_reply

    # -----------------------------------------------------------------------
    # 단계 2: 커리큘럼 생성 (generating) — 프론트엔드가 직접 이 phase로 진입 시
    # -----------------------------------------------------------------------
    elif state.phase == "generating":
        active_agent = "curriculum_agent"
        logger.info("[orchestrator] phase=generating regen=%s", state.regen_count)

        state, reply, final_curriculum, active_agent = _run_generating_chain(
            state, messages, vectorstore, curriculum_mod, evaluator_mod,
        )

    # -----------------------------------------------------------------------
    # 단계 3: 평가 (evaluating) — 프론트엔드가 직접 이 phase로 진입 시
    # -----------------------------------------------------------------------
    elif state.phase == "evaluating":
        active_agent = "evaluator"
        logger.info("[orchestrator] phase=evaluating")

        if state.curriculum:
            state, reply, final_curriculum, active_agent = _run_evaluating(
                state, messages, vectorstore, curriculum_mod, evaluator_mod,
            )
        else:
            logger.error("[orchestrator] evaluating phase but no curriculum in state")
            reply = "오류: 평가할 커리큘럼이 없습니다. 처음부터 다시 시작해 주세요."
            state.phase = "collecting"
            active_agent = "info_collector"

    # -----------------------------------------------------------------------
    # 단계 4: 완료 (complete)
    # -----------------------------------------------------------------------
    elif state.phase == "complete":
        active_agent = "complete"
        reply = "커리큘럼이 이미 완성되어 있습니다. 수정이 필요하시면 말씀해 주세요."
        final_curriculum = state.curriculum

    return {
        "reply": reply,
        "complete": state.phase == "complete",
        "curriculum": final_curriculum,
        "state": state.model_dump(),
        "active_agent": active_agent,
    }


# ---------------------------------------------------------------------------
# 내부 전이 함수
# ---------------------------------------------------------------------------

def _run_generating_chain(state, messages, vectorstore, curriculum_mod, evaluator_mod):
    """
    generating 단계를 실행하고 즉시 evaluating 단계로 전이한다.
    재생성 횟수가 초과되면 현재 커리큘럼을 완료 처리한다.
    재생성된 커리큘럼이 이전과 유사하면 정보 추가 수집으로 라우팅한다.
    """
    feedback = state.last_evaluator_feedback

    # 재생성 횟수 초과
    if state.regen_count >= state.max_regen:
        logger.warning("[orchestrator] max_regen(%s) reached → force complete", state.max_regen)
        state.phase = "complete"
        reply = (
            f"**최대 재생성 횟수({state.max_regen}회)에 도달**했습니다.\n"
            "현재 커리큘럼을 최종 결과로 제공합니다. 아래에서 확인하세요."
        )
        return state, reply, state.curriculum, "complete"

    regen_label = f" (재생성 {state.regen_count}차)" if state.regen_count > 0 else ""
    is_regen = state.regen_count > 0

    # 재생성 직전 커리큘럼을 저장 (유사도 비교용)
    state.previous_curriculum = state.curriculum

    logger.info("[orchestrator] running curriculum_agent regen=%s", state.regen_count)
    result = curriculum_mod.run_curriculum_agent(
        state.collected_info,
        vectorstore,
        evaluator_feedback=feedback,
        regen_count=state.regen_count,
    )
    curriculum = result.get("curriculum")

    state.agent_log.append(
        f"[커리큘럼 생성 에이전트{regen_label}] {'완료' if curriculum else '실패'}"
    )

    if not curriculum:
        reply = (
            f"**📝 커리큘럼 생성 에이전트{regen_label}**: 커리큘럼 생성에 실패했습니다.\n"
            + (result.get("reply") or "다시 시도해 주세요.")
        )
        return state, reply, None, "curriculum_agent"

    # ------------------------------------------------------------------
    # 재생성인 경우: 이전 커리큘럼과 유사도 검사
    # ------------------------------------------------------------------
    if is_regen:
        similarity = _calculate_similarity(state.previous_curriculum, curriculum)
        logger.info("[orchestrator] curriculum similarity=%.2f (threshold=%.2f)", similarity, SIMILARITY_THRESHOLD)

        if similarity >= SIMILARITY_THRESHOLD:
            # 유사 → 추가 정보 수집으로 라우팅
            state.curriculum = curriculum  # 일단 저장은 해둠
            state.phase = "collecting"
            state.similarity_context = (
                f"커리큘럼 생성 에이전트가 {state.regen_count}차 재생성을 시도했지만 "
                f"이전 커리큘럼과 {similarity:.0%} 유사한 결과를 반복 생성했습니다.\n"
                f"평가 에이전트의 개선 요청: {feedback}"
            )
            state.agent_log.append(
                f"[오케스트레이터] 유사도 {similarity:.0%} 감지 → 정보 추가 수집으로 전환"
            )

            reply = (
                f"**📝 커리큘럼 생성 에이전트{regen_label}**: 재생성을 완료했지만, "
                f"이전 커리큘럼과 **{similarity:.0%} 유사**한 결과가 반복되고 있습니다.\n\n"
                "커리큘럼의 방향을 바꾸려면 **더 구체적인 정보**가 필요합니다.\n"
                "**📋 정보 수집 에이전트**가 추가 요구사항을 여쭤볼게요.\n\n"
                "---"
            )
            return state, reply, curriculum, "info_collector"

    # ------------------------------------------------------------------
    # 정상 진행: 평가 단계로 전이
    # ------------------------------------------------------------------
    state.curriculum = curriculum
    state.phase = "evaluating"
    state.similarity_context = ""  # 유사도 컨텍스트 초기화

    eval_transition = (
        f"\n**📝 커리큘럼 생성 에이전트{regen_label}**: 커리큘럼 생성이 완료되었습니다.\n"
        "**🔍 평가 에이전트**로 전달해 품질을 검토합니다...\n\n"
    )

    state, eval_reply, final_curriculum, active_agent = _run_evaluating(
        state, messages, vectorstore, curriculum_mod, evaluator_mod,
    )

    return state, eval_transition + eval_reply, final_curriculum, active_agent


def _run_evaluating(state, messages, vectorstore, curriculum_mod, evaluator_mod):
    """
    evaluating 단계를 실행한다.
    통과 시 complete, 미통과 시 generating을 재귀 호출한다.
    """
    attempt = state.regen_count + 1  # 1-based 표시용

    eval_result = evaluator_mod.run_evaluator(state.curriculum, state.collected_info)

    passed     = eval_result["passed"]
    feedback   = eval_result["feedback"]
    summary    = eval_result["reply"]
    evaluation = eval_result.get("evaluation", {})

    # 평가 이력 저장
    state.evaluation_history.append({
        "attempt": attempt,
        "passed": passed,
        "summary": summary,
        "feedback": feedback,
        **{k: evaluation.get(k) for k in _EVAL_LABELS},
    })
    state.agent_log.append(
        f"[평가 에이전트 {attempt}차] {'통과' if passed else '미통과'} — {summary[:60]}"
    )
    logger.info("[orchestrator] evaluation attempt=%s passed=%s", attempt, passed)

    eval_lines = _eval_summary_lines(evaluation)

    if passed:
        state.phase = "complete"
        reply = (
            f"**🔍 평가 에이전트 ({attempt}차 검토)**\n\n"
            f"{eval_lines}\n\n"
            f"**결과: ✅ 평가 통과**\n{summary}\n\n"
            "커리큘럼이 확정되었습니다. 아래에서 확인하세요."
        )
        return state, reply, state.curriculum, "complete"

    # 미통과
    state.last_evaluator_feedback = feedback
    state.regen_count += 1
    state.phase = "generating"
    logger.info("[orchestrator] evaluation failed → regen #%s", state.regen_count)

    if state.regen_count > state.max_regen:
        state.phase = "complete"
        reply = (
            f"**🔍 평가 에이전트 ({attempt}차 검토)**\n\n"
            f"{eval_lines}\n\n"
            f"**결과: ❌ 평가 미통과**\n{summary}\n\n"
            f"최대 재생성 횟수({state.max_regen}회)에 도달했습니다. "
            "현재 커리큘럼을 최종 결과로 제공합니다."
        )
        return state, reply, state.curriculum, "complete"

    # 재생성 안내 + 재귀 호출
    regen_notice = (
        f"**🔍 평가 에이전트 ({attempt}차 검토)**\n\n"
        f"{eval_lines}\n\n"
        f"**결과: ❌ 평가 미통과**\n{summary}\n\n"
        f"**개선 요청사항:**\n{feedback}\n\n"
        f"---\n**📝 커리큘럼 생성 에이전트 ({state.regen_count}차 재생성)**으로 전달합니다...\n\n"
    )

    state, regen_reply, final_curriculum, active_agent = _run_generating_chain(
        state, messages, vectorstore, curriculum_mod, evaluator_mod,
    )
    return state, regen_notice + regen_reply, final_curriculum, active_agent
