BEGIN;
-- 제공 안전장치: 아래 DO 구문은 수정 과제가 아닙니다.
DO $guard$
BEGIN
    IF current_database() <> 'day19_build_h' THEN
        RAISE EXCEPTION 'Wrong database: expected day19_build_h, got %', current_database();
    END IF;
END;
$guard$;
CREATE TABLE public.categories (
    -- TODO H-DDL-01: categories.id의 BIGINT·ALWAYS IDENTITY·PK를 선언하세요.
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- TODO H-DDL-02: categories.name의 VARCHAR(30)·NOT NULL·UNIQUE를 선언하세요.
    name VARCHAR(30) NOT NULL UNIQUE,
    -- TODO H-DDL-03: categories_name_length CHECK를 작성하세요.
    CONSTRAINT categories_name_length
        CHECK (char_length(name) BETWEEN 1 AND 30)
);
COMMIT;
