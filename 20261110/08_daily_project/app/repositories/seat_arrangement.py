import json
from pathlib import Path
from threading import Lock
from datetime import date

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "seat_arrangement.json"
LOCK = Lock()

def read_data() -> dict:
    # 파일이 없거나 JSON이 손상되면 예외를 알립니다. 빈 목록으로 덮어쓰지 않습니다.
    with DATA.open("r", encoding="utf-8") as f:
        return json.load(f)

def seat_arrangement_list(ymd: date | None = None) -> dict | list:
    with LOCK:
        data = read_data()
    if ymd is not None:
        # json 객체의 키는 문자열이므로 변환 필요
        return data[ymd.isoformat()]
    return data