from sqlalchemy.orm import Session
from app.models import Category

class CategoryRepository:
    # 자바로 치면 this 를 기재한 거라 생각하면 이해가 편함
    def __init__(self, db: Session):
        self.db = db;

    def create_category(self, name: str) -> Category:
        category = Category(name=name)
        self.db.add(category)
        self.db.flush()
        return category

    def get_or_create_category(self, name: str) -> Category:
        category = self.db.scalar(
            select(Category).where(Category.name == name)
        )

        if category is None:
            category = self.create_category(name)  # db 전달 불필요

        return category