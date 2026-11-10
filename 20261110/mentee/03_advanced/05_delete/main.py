"""HTTP와 저장 책임을 구분하는 지식 관리 API입니다."""
from fastapi import FastAPI, HTTPException, Response
import logic as repo
from http import HTTPStatus
from schemas import Knowledge, KnowledgeWrite

app = FastAPI(title="Knowledge API")

@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):
    return repo.list_entries(category)

@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    entry = repo.find_entry(knowledge_id)
    if entry is None:
        raise HTTPException(404, "Knowledge not found")
    return entry

@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    return repo.create(payload.model_dump())

@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    updated = repo.update(knowledge_id, payload.model_dump())
    if updated is None:
        raise HTTPException(404, "Knowledge not found")
    return updated

@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    deleted = repo.delete(knowledge_id)
    if not deleted:
        raise HTTPException(404, "Knowledge not found")
    # TODO H05-02: 삭제 성공 시 본문 없는 204를 반환하세요.
    
    # [주의]
    # 이 코드 자체가 응답 상태를 204로 설정하지는 않습니다.
    # 일반적인 FastAPI 기본 설정에서는 200 OK + 응답 본문 204 가 됨
    #return HTTPStatus.NO_CONTENT

    # 데코레이터의 'status_code=204' 이 설정 때문에, 204 No Content 가 나갔던 것
    # 데코레이터를 쓸거면 차라리 return 만 하면 본문 없는 204 반환됨

    return Response(status_code=204)
