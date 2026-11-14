import json
import pytest
from http import HTTPStatus

"""HTTP 입력·응답과 저장 결과를 실제 임시 파일로 확인합니다."""
def test_health(isolated_client):
    client, _ = isolated_client
    r = client.get("/health")

    # TODO H01-02: r의 HTTP 상태가 200인지 assert로 검사하세요.
    assert r.status_code == HTTPStatus.OK, ""

    # TODO H01-03: r의 JSON 본문이 정확히 status=ok인지 검사하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    raise NotImplementedError("H01-03를 먼저 완성하세요.")
    # END H01-03
