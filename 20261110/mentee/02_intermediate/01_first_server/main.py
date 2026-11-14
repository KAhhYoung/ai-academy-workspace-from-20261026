"""내 PC에서 첫 요청 성공시키기. 실행: python -m uvicorn main:app --reload"""
from fastapi import FastAPI, HTTPException, Response

# BEGIN A01-01
# TODO A01-01: FastAPI 앱의 title을 Knowledge API로, description을 AI 플랫폼의 원본 지식 관리 연습으로 설정하세요.
app = FastAPI(
    title="Knowledge API",
    description="AI 플랫폼의 원본 지식 관리 연습"
)

@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    # TODO A01-02: GET /health에서 status=ok인 dict를 반환하세요.
    return {"status": "ok"}

@app.get("/hello")
def hello(name: str = "수강생"):
    # TODO A01-03: name을 포함한 안녕하세요, {name} 문자열을 message 값으로 반환하세요.
    return f'안녕하세요, {name}'
