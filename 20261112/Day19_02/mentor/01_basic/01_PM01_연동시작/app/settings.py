from functools import lru_cache
from sqlalchemy.engine import URL
from pydantic_settings import BaseSettings, SettingsConfigDict

# 전역변수는 파일(모듈)에 속하고, 클래스 변수는 클래스에 속함
class Settings(BaseSettings):
    # 설정 필드: 환경변수나 .env에서 읽고, 값이 없으면 기본값 사용
    # BaseSettings 를 상속했기 때문에 아래 변수들은 Pydantic이 관리하는 설정 필드. 인스턴스를 만들 때 .env 등의 설정을 읽어서 값이 정해짐
    # 이 코드의 db_user는 인스턴스마다 값을 가지는 설정 필드. 클래스 변수보다는 인스턴스 변수로 이해하면 됨
    db_user: str = "postgres"
    db_password: str = "CHANGE_ME"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "day19_crud_b"
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 강사 해설: 외부 API와 달리 DB 연결 정보는 비밀 설정입니다. URL.create가 비밀번호 인코딩을 맡게 합니다.
    @property
    def database_url(self) -> URL:
        # 비밀번호를 URL 문자열로 직접 이어 붙이지 않고 URL.create가 인코딩을 맡습니다.
        return URL.create(
            "postgresql+psycopg",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

@lru_cache
def get_settings() -> Settings:
    return Settings()
