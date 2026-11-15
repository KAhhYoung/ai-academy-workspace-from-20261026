from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from app.db import get_engine
from app.errors import DataConflictError, KnowledgeNotFoundError
from app.routers.knowledge import router

# 강사 해설: Day17의 URL·Method·입력·응답 계약을 유지하고 저장소만 PostgreSQL로 교체합니다.
app = FastAPI(title="Knowledge API · PostgreSQL · Advanced")
app.include_router(router)

@app.exception_handler(KnowledgeNotFoundError)
async def not_found_handler(_, exc: KnowledgeNotFoundError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"detail":"Knowledge not found", "knowledge_id":exc.knowledge_id})

@app.exception_handler(DataConflictError)
async def conflict_handler(_, exc: DataConflictError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409, content={"detail":exc.message})

@app.get("/health")
def health():
    try:
        with get_engine().connect() as conn: conn.execute(text("SELECT 1"))
        return {"status":"ok", "database":"reachable"}
    except OperationalError as exc:
        raise HTTPException(503, "Database unavailable") from exc

    # ---------------------------
    # OperationalError
    #  DB에 연결하거나 DB 작업을 수행하는 과정에서 발생하는 운영상 오류. SQLAlchemy에서는 DB 드라이버가 보고한 오류를 이 예외로 감싸서 전달
    #  대표적인 원인
    #  - PostgreSQL 서버가 실행되지 않음
    #  - 호스트나 포트가 잘못됨
    #  - 사용자 이름이나 비밀번호가 잘못됨
    #  - 접속하려는 DB가 존재하지 않음
    #  - 작업 중 DB 연결이 끊어짐
    # ---------------------------
