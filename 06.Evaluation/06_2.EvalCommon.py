from __future__ import annotations

import importlib.util
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_RAG_DIR = os.path.join(ROOT_DIR, "05.Advanced_RAG")
DEFAULT_TESTSET_PATH = os.path.join(ROOT_DIR, "06.Evaluation", "06_8.TestsetTemplate.json")

# 평가 스크립트도 서비스와 같은 환경변수를 써야 하므로, 프로젝트 루트의 .env를 먼저 읽는다.
load_dotenv(os.path.join(ROOT_DIR, ".env"))

_MODULES = None
_VECTORSTORE = None
_CHAIN = None


def _load_module(module_name: str, filename: str):
    # 숫자가 들어간 파일명은 일반 import가 어려워서, 파일 경로 기준으로 직접 불러온다.
    path = os.path.join(ADVANCED_RAG_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_project_modules():
    # 평가 코드에서도 본 서비스 코드와 같은 스키마/검색 로직을 재사용하기 위한 진입점이다.
    global _MODULES
    if _MODULES is not None:
        return _MODULES

    schemas = _load_module("schemas", "05_2.Schemas.py")
    indexing = _load_module("indexing", "05_4.Indexing.py")
    retrieval = _load_module("retrieval", "05_5.Retrieval.py")

    _MODULES = SimpleNamespace(
        schemas=schemas,
        indexing=indexing,
        retrieval=retrieval,
        Message=schemas.Message,
        CollectedInfo=schemas.CollectedInfo,
        CurriculumPlan=schemas.CurriculumPlan,
    )
    return _MODULES


def load_test_cases(path: str | None = None) -> list[dict[str, Any]]:
    # 테스트셋은 {"cases": [...]} 또는 바로 리스트 형태 둘 다 읽을 수 있게 한다.
    testset_path = path or DEFAULT_TESTSET_PATH
    with open(testset_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, dict):
        return list(raw.get("cases", []))
    if isinstance(raw, list):
        return raw
    raise ValueError("Testset must be a list or an object with a 'cases' field.")


def get_vectorstore(modules=None):
    # 인덱싱은 무거우므로 평가 중에는 한 번만 열고 계속 재사용한다.
    global _VECTORSTORE
    modules = modules or load_project_modules()
    if _VECTORSTORE is None:
        _VECTORSTORE = modules.indexing.setup_vector_store()
    return _VECTORSTORE


def get_chain(modules=None, vectorstore=None):
    # 최종 커리큘럼 생성 체인도 한 번만 만들어 여러 케이스에서 함께 쓴다.
    global _CHAIN
    modules = modules or load_project_modules()
    vectorstore = vectorstore or get_vectorstore(modules)
    if _CHAIN is None:
        _CHAIN = modules.retrieval.build_chain(vectorstore)
    return _CHAIN


def build_groups(info) -> dict[str, dict[str, Any]]:
    # 본 서비스와 같은 그룹 규칙을 평가 코드에서도 그대로 사용한다.
    return {
        "group_a": {
            "name": "그룹 A",
            "types": ["균형형", "이해형"],
            "count": info.count_balanced + info.count_learner,
        },
        "group_b": {
            "name": "그룹 B",
            "types": ["과신형", "실행형"],
            "count": info.count_overconfident + info.count_doer,
        },
        "group_c": {
            "name": "그룹 C",
            "types": ["판단형", "조심형"],
            "count": info.count_analyst + info.count_cautious,
        },
    }


def build_case_inputs(case: dict[str, Any], modules=None):
    # 테스트 케이스 하나를 실제 generate 입력 형식으로 바꿔 준다.
    modules = modules or load_project_modules()
    info = modules.CollectedInfo(**case["collected_info"])
    groups = build_groups(info)
    raw_messages = [modules.Message(**message) for message in case.get("messages", [])]
    conversation = [SystemMessage(content=modules.retrieval.COLLECTION_SYSTEM_PROMPT)] + modules.retrieval.to_lc_messages(raw_messages)
    return info, groups, conversation


def generate_plan(case: dict[str, Any], modules=None, vectorstore=None, chain=None):
    # 평가할 커리큘럼을 실제 서비스와 같은 체인으로 생성한다.
    modules = modules or load_project_modules()
    vectorstore = vectorstore or get_vectorstore(modules)
    chain = chain or get_chain(modules, vectorstore)
    info, groups, conversation = build_case_inputs(case, modules)
    return chain.invoke(
        {
            "conversation": conversation,
            "info": info,
            "groups": groups,
        }
    )


def collect_reference_bundle(case: dict[str, Any], modules=None, vectorstore=None) -> dict[str, str]:
    # Faithfulness 평가는 생성 당시 참고한 문맥을 같이 봐야 하므로, 참조 문서를 다시 모아 둔다.
    modules = modules or load_project_modules()
    vectorstore = vectorstore or get_vectorstore(modules)
    info, groups, _conversation = build_case_inputs(case, modules)
    corpus_cache: dict[str, Any] = {}

    ctx_a = modules.retrieval.retrieve_group_context(
        vectorstore,
        groups["group_a"]["types"],
        group_label="group_a",
        corpus_cache=corpus_cache,
    )
    ctx_b = modules.retrieval.retrieve_group_context(
        vectorstore,
        groups["group_b"]["types"],
        group_label="group_b",
        corpus_cache=corpus_cache,
    )
    ctx_c = modules.retrieval.retrieve_group_context(
        vectorstore,
        groups["group_c"]["types"],
        group_label="group_c",
        corpus_cache=corpus_cache,
    )
    curriculum_examples = modules.retrieval.retrieve_curriculum_examples(
        vectorstore,
        info,
        corpus_cache=corpus_cache,
    )

    combined_text = "\n\n".join(
        [
            "[AX Compass - Group A]",
            ctx_a,
            "[AX Compass - Group B]",
            ctx_b,
            "[AX Compass - Group C]",
            ctx_c,
            "[Curriculum Examples]",
            curriculum_examples,
        ]
    )
    return {
        "group_a": ctx_a,
        "group_b": ctx_b,
        "group_c": ctx_c,
        "curriculum_examples": curriculum_examples,
        "combined_text": combined_text,
    }


def plan_to_text(plan) -> str:
    # Judge 모델에 넘길 때는 사람이 읽기 쉬운 JSON 문자열로 바꿔서 전달한다.
    return json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)


def save_report(report: dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
