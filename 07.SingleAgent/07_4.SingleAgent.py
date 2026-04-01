from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _load_agent_helpers():
    # 번호가 붙은 파일명은 일반 import가 어려워서, 같은 폴더 파일을 경로로 직접 불러온다.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_3.AgentHelpers.py")
    spec = importlib.util.spec_from_file_location("agent_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_helpers"] = module
    spec.loader.exec_module(module)
    return module


_agent_helpers = _load_agent_helpers()
ROOT_DIR = _agent_helpers.ROOT_DIR
load_agent_context = _agent_helpers.load_agent_context


class SingleCurriculumAgent:
    # 하나의 에이전트가 수집, 생성, 평가까지 순서대로 처리하는 예시 클래스다.
    def __init__(self):
        self.ctx = load_agent_context()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = self.ctx.indexing.setup_vector_store()
        self.chain = self.ctx.retrieval.build_chain(self.vectorstore)

    def _ensure_collected_info(self, agent_input):
        # 이미 구조화된 정보가 있으면 그대로 쓰고, 없으면 수집 프롬프트로 다시 추출한다.
        if agent_input.collected_info:
            return self.ctx.schemas.CollectedInfo(**agent_input.collected_info)

        raw_messages = [self.ctx.schemas.Message(**message) for message in agent_input.messages]
        lc_messages = [SystemMessage(content=self.ctx.retrieval.COLLECTION_SYSTEM_PROMPT)] + self.ctx.retrieval.to_lc_messages(raw_messages)
        reply = self.llm.invoke(lc_messages).content
        extract_llm = self.llm.with_structured_output(self.ctx.schemas.CollectedInfo)
        info = extract_llm.invoke(
            lc_messages
            + [AIMessage(content=reply)]
            + [HumanMessage(content="위 대화에서 수집한 모든 정보를 구조화해서 추출해줘.")]
        )
        return info

    def _generate_plan(self, info, agent_input):
        # generate API와 같은 입력 형식으로 conversation과 groups를 만든 뒤 체인을 호출한다.
        raw_messages = [self.ctx.schemas.Message(**message) for message in agent_input.messages]
        conversation = [SystemMessage(content=self.ctx.retrieval.COLLECTION_SYSTEM_PROMPT)] + self.ctx.retrieval.to_lc_messages(raw_messages)
        groups = self.ctx.eval_common.build_groups(info)
        return self.chain.invoke(
            {
                "conversation": conversation,
                "info": info,
                "groups": groups,
            }
        )

    def _build_eval_case(self, case_id: str, info, agent_input) -> dict:
        # 평가 모듈은 test case 형식을 기대하므로, Single Agent 입력을 그 형식으로 변환한다.
        return {
            "case_id": case_id,
            "messages": agent_input.messages,
            "collected_info": info.model_dump(),
            "expected": {
                "must_cover": agent_input.must_cover,
                "retrieval": {},
            },
        }

    def _evaluate(self, case: dict, plan) -> dict:
        # 06.Evaluation의 네 가지 평가를 한 번에 호출한다.
        retrieval_result = self.ctx.retrieval_eval.evaluate_case(case, self.ctx.eval_common.load_project_modules(), self.vectorstore)
        rule_result = self.ctx.rule_eval.evaluate_case(case, plan, self.ctx.eval_common.load_project_modules())
        references = self.ctx.eval_common.collect_reference_bundle(case, self.ctx.eval_common.load_project_modules(), self.vectorstore)
        faithfulness_result = self.ctx.faithfulness_eval.evaluate_case(case, plan, references)
        coverage_result = self.ctx.coverage_eval.evaluate_case(case, plan)
        return {
            "retrieval": retrieval_result,
            "rule_based": rule_result,
            "faithfulness": faithfulness_result,
            "requirement_coverage": coverage_result,
        }

    def _default_output_path(self, case_id: str) -> str:
        # 결과를 여러 번 남길 수 있도록, 시각 기반 파일명을 자동으로 만든다.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(ROOT_DIR, "07.SingleAgent", f"{stamp}_{case_id}_SingleAgentReport.md")

    def _render_markdown(self, case_id: str, info, plan, evaluation: dict | None) -> str:
        # 발표나 점검용으로 바로 읽을 수 있는 한국어 md 리포트를 만든다.
        lines: list[str] = []
        lines.append(f"# Single Agent 결과 리포트")
        lines.append("")
        lines.append(f"- 케이스 ID: `{case_id}`")
        lines.append(f"- 회사명: `{info.company_name}`")
        lines.append(f"- 주제: `{info.topic}`")
        lines.append(f"- 총 교육시간: `{info.days * info.hours_per_day}시간`")
        lines.append("")
        lines.append("## 수집된 요구사항")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(info.model_dump(), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("## 생성된 커리큘럼")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
        lines.append("```")

        if evaluation:
            lines.append("")
            lines.append("## 평가 결과 요약")
            lines.append("")
            lines.append("| 항목 | 점수 |")
            lines.append("|---|---:|")
            lines.append(f"| Retrieval Precision@k | {evaluation['retrieval']['overall']['average_precision_at_k']:.2f} |")
            lines.append(f"| Faithfulness | {evaluation['faithfulness']['score']}/5 |")
            lines.append(f"| Requirement Coverage | {evaluation['requirement_coverage']['score']}/5 |")
            lines.append(f"| Rule-based Validation | {evaluation['rule_based']['score']:.2f} |")
            lines.append("")
            lines.append("### Faithfulness 종합 의견")
            lines.append("")
            lines.append(f"- {evaluation['faithfulness']['summary']}")
            lines.append("")
            lines.append("### Requirement Coverage 종합 의견")
            lines.append("")
            lines.append(f"- {evaluation['requirement_coverage']['summary']}")
        else:
            lines.append("")
            lines.append("> 이번 실행에서는 평가를 생략했다.")

        lines.append("")
        return "\n".join(lines)

    def run(self, agent_input):
        # Single Agent의 전체 흐름을 수행한다.
        info = self._ensure_collected_info(agent_input)
        plan = self._generate_plan(info, agent_input)
        case = self._build_eval_case(agent_input.case_id, info, agent_input)
        evaluation = self._evaluate(case, plan) if agent_input.run_evaluation else None
        output_path = agent_input.output_path or self._default_output_path(agent_input.case_id)

        markdown = self._render_markdown(agent_input.case_id, info, plan, evaluation)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(markdown)

        return {
            "case_id": agent_input.case_id,
            "collected_info": info.model_dump(),
            "curriculum_plan": plan.model_dump(),
            "evaluation": evaluation,
            "markdown_path": output_path,
        }
