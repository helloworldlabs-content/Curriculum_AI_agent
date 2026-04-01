from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_RAG_DIR = os.path.join(ROOT_DIR, "05.Advanced_RAG")
EVALUATION_DIR = os.path.join(ROOT_DIR, "06.Evaluation")

# Single Agent도 서비스와 같은 API 키와 환경변수를 그대로 사용한다.
load_dotenv(os.path.join(ROOT_DIR, ".env"))

_CONTEXT = None


def _load_module(alias: str, path: str):
    # 숫자가 포함된 파일명은 일반 import가 어려워서, 파일 경로 기준으로 직접 불러온다.
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def load_agent_context():
    # Advanced RAG와 Evaluation 코드를 한 번만 불러서 재사용한다.
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT

    schemas = _load_module("schemas", os.path.join(ADVANCED_RAG_DIR, "05_2.Schemas.py"))
    indexing = _load_module("indexing", os.path.join(ADVANCED_RAG_DIR, "05_4.Indexing.py"))
    retrieval = _load_module("retrieval", os.path.join(ADVANCED_RAG_DIR, "05_5.Retrieval.py"))
    eval_common = _load_module("eval_common", os.path.join(EVALUATION_DIR, "06_2.EvalCommon.py"))
    retrieval_eval = _load_module("retrieval_eval", os.path.join(EVALUATION_DIR, "06_3.RetrievalEval.py"))
    faithfulness_eval = _load_module("faithfulness_eval", os.path.join(EVALUATION_DIR, "06_4.FaithfulnessEval.py"))
    coverage_eval = _load_module("coverage_eval", os.path.join(EVALUATION_DIR, "06_5.CoverageEval.py"))
    rule_eval = _load_module("rule_eval", os.path.join(EVALUATION_DIR, "06_6.RuleEval.py"))

    _CONTEXT = SimpleNamespace(
        schemas=schemas,
        indexing=indexing,
        retrieval=retrieval,
        eval_common=eval_common,
        retrieval_eval=retrieval_eval,
        faithfulness_eval=faithfulness_eval,
        coverage_eval=coverage_eval,
        rule_eval=rule_eval,
    )
    return _CONTEXT
