from pydantic import BaseModel, Field
from datetime import date

'''
    저장은
    {
        "2026-11-10": [],
        "2026-11-11": []
    }

    출력은
    {
        "ymd": 2026-11-10,
        "arrange": []
    }
'''

# 입력모델
class SeatArrangementCreate(BaseModel):
    ymd: date = Field(
        ge=date(2026, 1, 1),
        le=date(2026, 12, 31),
    )

# 출력모델
class SeatArrangement(SeatArrangementCreate):
    arrange: list