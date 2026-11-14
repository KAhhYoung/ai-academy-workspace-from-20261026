from fastapi import APIRouter
from datetime import date
from app.schemas import SeatArrangement, SeatArrangementCreate
from app.repositories import seat_arrangement

router = APIRouter(prefix="/api/seats", tags=["좌석 배치"])

@router.get("", response_model=list[SeatArrangement])
# ymd: 허용 타입(date | None) = 기본값(None)
def list_seat_arrangement(ymd: date | None = None):
    pass

@router.post("", status_code=201, response_model=SeatArrangement)
def create_seat_arrangement(payload: SeatArrangementCreate):
    pass

@router.delete("/{ymd}", status_code=204)
    pass