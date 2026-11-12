from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Identity, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# PostgreSQL에서는 BIGINT, SQLite 기반 정적 검수에서는 INTEGER가 되도록 정의합니다.
# 기본적으로 Integer를 사용하되, PostgreSQL에서는 BigInteger를 사용한다는 SQLAlchemy 설정
ID_TYPE = Integer().with_variant(BigInteger, "postgresql")

class Base(DeclarativeBase):
    pass

# DeclarativeBase 를 상속하는 이유: 클래스를 DB 테이블과 연결하는 SQLAlchemy 모델로 만들기 위해서.
# 강사 해설: ORM Model은 DB Table 계약을 Python에서 표현합니다. Pydantic Schema와 역할이 다릅니다.
# 같은 Base를 상속한 모델들의 테이블 정의는 Base.metadata에 모임. Base.metadata 로 접근
'''
- Mapped[int]: SQLAlchemy가 관리하는 정수형 속성
- ID_TYPE: 앞서 정의한 DB별 정수 타입. PostgreSQL에서는 BIGINT
- Identity(always=True): DB가 ID를 자동 생성. 직접 값을 지정하는 것은 기본적으로 제한
- primary_key=True: 행을 구분하는 기본키

- knowledge_items:
    카테고리에서 소속 글 목록을 조회할 필요가 없다면 → knowledge_items를 생략해도 됩니다.
    category.knowledge_items로 글 목록에 접근하고 싶다면 → 관계를 정의합니다.
    보통은 FK를 기준으로 관계를 설정해요.
    FK 없이도 연결 조건을 직접 지정하면 relationship()을 사용할 수 있지만, 현재 코드에서는 FK와 relationship()을 함께 쓰는 방식으로 이해하면 됩니다.
    반대로 FK만 있고 knowledge_items는 없어도 됩니다. 다만 반대편에 back_populates="knowledge_items"가 있다면 그 설정도 함께 수정해야 해요.
'''
class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (
        CheckConstraint("char_length(name) BETWEEN 1 AND 30", name="categories_name_length"),
    )
    id: Mapped[int] = mapped_column(ID_TYPE, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(30), nullable=False, unique=True)
    knowledge_items: Mapped[list["Knowledge"]] = relationship(back_populates="category")

# 강사 해설: category_id는 실제 FK이고 relationship은 Python 객체 탐색을 돕는 ORM 기능입니다.
class Knowledge(Base):
    __tablename__ = "knowledge"
    __table_args__ = (
        CheckConstraint("char_length(title) BETWEEN 1 AND 80", name="knowledge_title_length"),
        CheckConstraint("char_length(content) BETWEEN 1 AND 2000", name="knowledge_content_length"),
    )
    id: Mapped[int] = mapped_column(ID_TYPE, Identity(always=True), primary_key=True)
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category_id: Mapped[int] = mapped_column(
        ID_TYPE, ForeignKey("categories.id", ondelete="RESTRICT", name="knowledge_category_fk"), nullable=False
    )
    category: Mapped[Category] = relationship(back_populates="knowledge_items")
