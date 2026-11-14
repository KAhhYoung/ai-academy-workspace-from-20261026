from sqlalchemy import select, insert
from sqlalchemy.orm import Session, joinedload
from app.models import Category, Knowledge

def list_knowledge(db: Session, category: str | None = None) -> list[Knowledge]:
    stmt = select(Knowledge).options(joinedload(Knowledge.category)).order_by(Knowledge.id)
    if category is not None:
        stmt = stmt.join(Knowledge.category).where(Category.name == category)
    return list(db.scalars(stmt).unique().all())

def get_knowledge(db: Session, knowledge_id: int) -> Knowledge | None:
    # TODO H06: PK 기준 상세 조회를 구현하세요.
    raise NotImplementedError("TODO H06")

def get_or_create_category(db: Session, name: str) -> Category:
    # TODO H07: 분류 재사용/생성과 flush를 구현하세요.
    # flush 를 해야 호출한 함수에서 id 사용 가능
    # [Q] flush() 하더라도 category.id 는 None 아닌가?
    # [A] 아니요. db.flush()가 성공하면 category.id에 DB가 생성한 ID가 자동으로 채워집니다. SQLAlchemy가 INSERT 결과를 같은 객체에 반영하기 때문입니다.
    stmt=select(Category).where(Category.name == name)
    category = db.scalar(stmt)
    if category is None:
        # 어차피 insert 문 만으로는 실행 안됨.. db.execute() or db.scalar()
        # add방식은 python ORM 객체를 생성 / insert문은 SQL 구문을 생성
        #category = insert(Category).values(name=name).returning(Category)
        category = Category(name=name)
        db.add(category)
        db.flush()
    return category
    
    

# *는 뒤에 오는 인자들을 반드시 이름을 붙여 전달하도록 만드는 구분자
def add_knowledge(db: Session, *, title: str, content: str, category_id: int) -> Knowledge:
    # TODO H08: 지식 ORM 객체를 만들고 Session에 등록하세요.
    raise NotImplementedError("TODO H08")

def delete_knowledge(db: Session, item: Knowledge) -> None:
    db.delete(item)
