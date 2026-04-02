from typing import Literal

from pydantic import BaseModel, Field


# 채팅 메시지 단위
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# 오케스트레이터 상태 — 요청/응답에 포함해 프론트엔드가 매 턴 전달
class OrchestratorState(BaseModel):
    phase: Literal["collecting", "generating", "evaluating", "complete"] = "collecting"
    collected_info: dict | None = None
    curriculum: dict | None = None
    previous_curriculum: dict | None = None   # 재생성 직전 커리큘럼 (유사도 비교용)
    regen_count: int = 0
    max_regen: int = 3
    last_evaluator_feedback: str = ""         # 평가 미통과 시 재생성에 사용
    similarity_context: str = ""              # 유사 커리큘럼 감지 시 InfoCollector에 전달
    evaluation_history: list[dict] = Field(default_factory=list)  # 평가 결과 이력
    agent_log: list[str] = Field(default_factory=list)


# /chat 요청/응답
class ChatRequest(BaseModel):
    messages: list[Message]
    state: OrchestratorState | None = None


class ChatResponse(BaseModel):
    reply: str
    complete: bool
    curriculum: dict | None = None
    state: OrchestratorState
    active_agent: str  # "info_collector" | "curriculum_agent" | "evaluator" | "complete"


# 커리큘럼 구성 단위
class Session(BaseModel):
    title: str
    duration_hours: float
    goals: list[str]
    activities: list[str]


class GroupInfo(BaseModel):
    group_name: str
    target_types: str
    participant_count: int
    focus_description: str


class GroupDaySessions(BaseModel):
    group_name: str
    sessions: list[Session]


class DaySchedule(BaseModel):
    day: int
    theme: str
    common_sessions: list[Session]
    group_sessions: list[GroupDaySessions]


# LLM이 최종 출력하는 커리큘럼 전체 구조
class CurriculumPlan(BaseModel):
    program_title: str
    target_summary: str
    groups: list[GroupInfo]
    daily_schedules: list[DaySchedule]
    expected_outcomes: list[str]
    notes: list[str]


# 에이전트가 submit_info 도구 호출 시 받는 수집 정보
class CollectedInfo(BaseModel):
    company_name:        str = Field(description="회사명 또는 팀 이름")
    goal:                str = Field(description="교육 목표")
    audience:            str = Field(description="교육 대상자")
    level:               str = Field(description="현재 AI 활용 수준")
    days:                int = Field(description="총 교육 기간 (일수)")
    hours_per_day:       int = Field(description="하루 교육 시간 (시간)")
    topic:               str = Field(description="원하는 핵심 주제")
    constraints:         str = Field(description="반영해야 할 조건 또는 제한사항")
    count_balanced:      int = Field(description="균형형 인원수")
    count_learner:       int = Field(description="이해형 인원수")
    count_overconfident: int = Field(description="과신형 인원수")
    count_doer:          int = Field(description="실행형 인원수")
    count_analyst:       int = Field(description="판단형 인원수")
    count_cautious:      int = Field(description="조심형 인원수")


# 평가 에이전트가 반환하는 구조화된 평가 결과
class EvaluationResult(BaseModel):
    passed: bool = Field(description="전체 평가 통과 여부")
    time_constraint_ok: bool = Field(description="하루 시간 합계가 hours_per_day와 일치하는지")
    feasibility_ok: bool = Field(description="세션 내용이 실제로 강의 가능한 수준인지")
    condition_fit_ok: bool = Field(description="기업 요구사항과 제약사항이 반영됐는지")
    group_balance_ok: bool = Field(description="3개 그룹의 세션 시간과 내용이 균형 잡혔는지")
    feedback: str = Field(description="커리큘럼 재생성 시 반영해야 할 구체적인 개선 요청사항")
    summary: str = Field(description="사용자에게 보여줄 평가 요약 (통과/미통과 이유 포함)")


# 인증 요청/응답
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
