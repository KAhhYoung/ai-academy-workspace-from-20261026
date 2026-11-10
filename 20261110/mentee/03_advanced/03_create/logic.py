"""데이터 처리 함수. HTTPException을 만들지 않고 dict·None·bool을 반환합니다."""
from storage import LOCK, read_data, write_data

def list_entries(category: str | None = None) -> list[dict]:
    with LOCK:
        data = read_data()
    entries = data["entries"]
    if category is not None:
        entries = [entry for entry in entries if entry["category"] == category]
    return entries


def find_entry(knowledge_id: int) -> dict | None:
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        if entry["id"] == knowledge_id:
            return entry
    return None

def create(fields: dict) -> dict:
    with LOCK:
        data = read_data()
        # TODO H03-01: fields를 복사한 entry를 만들고 현재 next_id를 id로 지정하세요.
        entry = fields.copy()
        entry["id"] = data["next_id"]
        # TODO H03-02: entry를 추가하고 다음 ID를 증가시켜 data 전체를 저장하세요.
        data["entries"].append(entry)
        data["next_id"] += 1

        # data 저장?
        write_data(data)
    return entry
