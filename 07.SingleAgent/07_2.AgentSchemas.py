from __future__ import annotations

from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    # messages는 대화형 입력을 그대로 넣을 때 사용한다.
    messages: list[dict] = Field(default_factory=list, description="대화 메시지 목록")
    # collected_info가 있으면 정보 수집 단계를 건너뛸 수 있다.
    collected_info: dict | None = Field(default=None, description="이미 구조화된 요구사항")
    # 평가에서 꼭 반영되길 기대하는 항목들이다.
    must_cover: list[str] = Field(default_factory=list, description="반드시 반영되길 기대하는 항목")
    # 평가를 같이 돌릴지 여부다.
    run_evaluation: bool = Field(default=True, description="평가까지 함께 실행할지 여부")
    # 결과 md 파일을 저장할 위치다. 비우면 자동으로 파일명이 만들어진다.
    output_path: str | None = Field(default=None, description="출력 markdown 경로")
    case_id: str = Field(default="single_agent_case", description="케이스 식별자")


class AgentOutput(BaseModel):
    case_id: str
    collected_info: dict
    curriculum_plan: dict
    evaluation: dict | None = None
    markdown_path: str
