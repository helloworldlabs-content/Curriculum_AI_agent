# -*- coding: utf-8 -*-
"""
AX(AI Transformation) 교육 커리큘럼 생성 챗봇
- 기업 맞춤형 AX 교육 커리큘럼 자동 생성
- OpenAI API 활용
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
SYSTEM_PROMPT = """당신은 20년 경력의 IT·AI 교육 전문가이자 교육 스타트업 대표입니다.

[전문 분야]
- 기업 AX(AI Transformation) 교육 설계
- 대상자별 맞춤 커리큘럼 개발
- 실무 중심의 AI 교육 프로그램 기획

[역할]
사용자가 요청하는 기업의 AX 교육 커리큘럼을 전문적으로 설계합니다.

[커리큘럼 설계 원칙]
1. 교육 대상자의 현재 수준과 역할에 맞게 설계
2. 이론보다 실무 적용 중심으로 구성
3. 단계별 학습으로 성취감과 실력 향상 동시 달성
4. 교육 후 즉시 업무에 활용 가능한 내용 포함
5. 기업의 업종/규모/목적에 최적화

[커리큘럼 출력 형식]
다음 형식으로 커리큘럼을 제공하세요:

## 📋 AX 교육 커리큘럼: [회사명/대상]

### 교육 개요
- **교육 대상**:
- **총 교육 시간**:
- **교육 목표**:

### 커리큘럼 구성
각 모듈별로:
**[모듈 번호]. [모듈명]** (소요 시간)
- 학습 목표
- 주요 내용
- 실습/실무 적용

### 기대 효과
- 수강 후 변화/성과

[대화 방식]
- 커리큘럼 생성에 필요한 정보가 부족하면 자연스럽게 추가 질문
- 필요한 핵심 정보: 교육 대상자, 업종, 교육 시간, 중점 주제/기능
- 한 번에 모든 정보를 묻지 말고 대화 흐름에 맞게 질문
- 친근하고 전문적인 어조 유지
"""

def chat(conversation_history: list[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        temperature=0.7,
    )
    return response.choices[0].message.content


def main():
    print("=" * 60)
    print("  AX 교육 커리큘럼 설계 챗봇")
    print("  AI Transformation 맞춤형 커리큘럼을 만들어드립니다")
    print("=" * 60)
    print("종료하려면 'q' 또는 'quit'을 입력하세요.\n")

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # 첫 인사
    greeting = chat(conversation_history + [
        {"role": "user", "content": "안녕하세요, 커리큘럼 상담 시작해주세요."}
    ])
    print(f"챗봇: {greeting}\n")
    conversation_history.append({"role": "assistant", "content": greeting})

    while True:
        user_input = input("나: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "종료", "exit"):
            print("\n챗봇: 감사합니다! 성공적인 AX 교육이 되길 바랍니다. 👋")
            break

        conversation_history.append({"role": "user", "content": user_input})

        response = chat(conversation_history)
        print(f"\n챗봇: {response}\n")

        conversation_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
