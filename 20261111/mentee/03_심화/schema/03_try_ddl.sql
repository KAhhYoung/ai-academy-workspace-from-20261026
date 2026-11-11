-- Day18 | 실습 03: 두 테이블·키·ERD | PPT 21~28, 69~70, 73 | 30분
-- SQL 편집: VS Code / 실행: pgAdmin Query Tool (주 경로), psql (선택 경로)
-- @step은 실행 단위를 표시하는 주석입니다. SQL 명령이 아닙니다.
-- 한 @step 구간씩 선택 실행합니다. 오류 뒤 ROLLBACK은 반드시 별도 실행합니다.
-- __TODO_...__는 구현할 빈칸, __NEW_ID__/__A_ID__/__B_ID__/__C_ID__는 실제 실행 입력입니다.
-- 입력 ID는 RETURNING에서 읽어 해당 토큰을 실제 정수로 바꿉니다. 4·5·6으로 예측하지 않습니다.
-- 아직 남은 토큰 때문에 생긴 문법 오류는 의도한 제약 실패로 인정하지 않습니다.

-- @step E3_GUARD expect=00000 capture=- reconnect=false
-- 제공된 안전 확인입니다. 이 블록은 TODO가 아닙니다.
-- 개인 수업용 DB에서만 실행합니다. 이름만 맞다고 운영 DB가 안전해지는 것은 아닙니다.
DO $guard$
BEGIN
  IF current_database() !~ '^knowledge_lab(_[a-z0-9_]+)?$' THEN
    RAISE EXCEPTION '개인 knowledge_lab DB에 다시 접속하세요. 현재 DB: %', current_database();
  END IF;
END;
$guard$;
SET search_path TO public;

-- @step E3_BEGIN expect=00000 capture=- reconnect=false
BEGIN;

-- @step E3_CREATE expect=00000 capture=- reconnect=false
CREATE TABLE public.categories_try (
-- TODO H03_01: 연습용 분류의 id를 행 식별 키로 선언하세요. IDENTITY 자동 발급은 이 축소 예제에서 사용하지 않습니다.
  id INTEGER NOT NULL PRIMARY KEY,

-- TODO H03_03: 연습용 분류명은 NULL 금지·중복 금지·1~30자로 선언하세요. 공백 제거 정책은 새로 넣지 않습니다.
-- VARCHAR: 앞뒤 공백도 길이에 포함됩니다.
-- BETWEEN 1 AND 30: 1,30 포함
-- VARCHAR(30) 은 자료형 정의방식, CHECK는 제약
  name VARCHAR(30) NOT NULL UNIQUE,
  CONSTRAINT ck_name_length
  CHECK (length(name) BETWEEN 1 AND 30)

);
CREATE TABLE public.knowledge_try (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
-- TODO H03_02: category_id가 categories_try.id를 참조하게 하세요. 자식이 부모를 가리키는 방향입니다.
-- 부모(categories_try)의 category.id는 기본 키(PRIMARY KEY) 또는 고유 키(UNIQUE)여야 합니다.
-- TODO H03_04: 부모 분류를 참조하는 지식이 있으면 부모 삭제를 거부하는 정책을 작성하세요.
  category_id INTEGER NOT NULL,
  CONSTRAINT fk_knowledge_category_id
    FOREIGN KEY (category_id)
    REFERENCES public.categories_try(id)
    ON DELETE RESTRICT
);

COMMIT;
