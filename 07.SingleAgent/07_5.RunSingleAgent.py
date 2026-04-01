from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def _load_local_module(alias: str, filename: str):
    # 번호가 포함된 파일명은 일반 import가 어려워서 파일 경로로 직접 불러온다.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_schemas = _load_local_module("agent_schemas", "07_2.AgentSchemas.py")
_agent_module = _load_local_module("single_agent", "07_4.SingleAgent.py")

AgentInput = _schemas.AgentInput
SingleCurriculumAgent = _agent_module.SingleCurriculumAgent


def main():
    parser = argparse.ArgumentParser(description="Run the single curriculum agent.")
    parser.add_argument("--input", required=True, help="Path to the agent input JSON file.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        payload = json.load(file)

    agent_input = AgentInput(**payload)
    agent = SingleCurriculumAgent()
    result = agent.run(agent_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
