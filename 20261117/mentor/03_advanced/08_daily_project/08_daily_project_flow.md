<08>
실행명령어: python ' ' (--action reuse or changed) --source live --allow-paid   / mode 는 기본값인 'fast' 사용
main.py
workflow.py - main
support.py - daily_project  (테스트질문: H01,H02,H03)
support.py - execute_case   (H01->streaming=True)
project.py - shared_limits (재시도 정책 설정)
support.py - load_settings
config.py - choose_model
support.py - execute_case --> case: cases.json, selected: knowledge_v1.json, instructions: instructions_v1.txt, question: cases.json
request.py - build_request --> 요청생성


결과저장: first_vs_final.json