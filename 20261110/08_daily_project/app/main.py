from fastapi import FastAPI
from app.routers import seat_arrangement

app = FastAPI(title="Seat Arrangement Table")
app.include_router(seat_arrangement.router)