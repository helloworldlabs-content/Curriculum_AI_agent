from typing import Literal

from pydantic import BaseModel, Field


# 채팅 메시지 단위
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# /chat 요청/응답
class ChatRequest(BaseModel):
    messages: list[Message]


class ChatResponse(BaseModel):
    reply: str
    complete: bool
    curriculum: dict | None = None


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


# 에이전트가 create_curriculum 도구 호출 시 받는 수집 정보
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


# 인증 요청/응답
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
