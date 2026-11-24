# Python 개발 환경 표준화 가이드 (VS Code + Ruff)
- by.김주영님

## 1. Ruff를 사용하는 이유

Ruff는 Python 코드를 검사하고 정리해 주는 도구입니다. 여러 도구를 따로 사용하지 않고도 다음 작업을 한곳에서 처리할 수 있습니다.

- **코드 포맷팅:** 들여쓰기, 공백, 줄바꿈 등 코드 모양을 통일합니다.
- **Import 정리:** 사용하지 않는 import를 찾고 import 순서를 정리합니다.
- **자동 수정:** Ruff가 안전하게 수정할 수 있는 코드 문제를 저장할 때 자동으로 고칩니다.
- **코드 검사:** 정해진 규칙에 어긋나는 부분을 알려줍니다.

팀원들이 같은 규칙을 사용하면 코드 스타일에 대한 불필요한 의견 교환이 줄고, 코드 리뷰와 PR Diff도 깔끔해집니다.

## 2. 설치 및 VS Code 설정

### ① Ruff 설치

1. VS Code에서 `Ctrl + Shift + X`를 누릅니다.
2. 검색창에 `Ruff`를 입력합니다.
3. **Ruff**를 설치합니다.

### ② settings.json 열기

1. `Ctrl + Shift + P`를 누릅니다.
2. `Preferences: Open User Settings (JSON)`을 검색해 실행합니다.
3. 열린 `settings.json`에 아래 설정을 추가합니다.

파일 위치:

```text
C:\Users\사용자_이름\AppData\Roaming\Code\User\settings.json
```

> 이미 `settings.json`에 다른 설정이 있다면 파일 전체를 덮어쓰지 말고, 아래 항목을 기존 설정에 합쳐 넣으세요. 같은 항목이 이미 있다면 중복으로 추가하지 말고 값을 조정하면 됩니다.

```jsonc
{
    // VS Code의 파일/폴더 아이콘 테마
    "workbench.iconTheme": "material-icon-theme",

    // GitHub Copilot 자동 제안 비활성화
    "github.copilot.enable": {
        "*": false,
        "plaintext": false,
        "markdown": false,
        "scminput": false
    },

    // Python 파일에만 적용되는 설정
    "[python]": {
        // 기본 포매터로 Ruff 사용
        "editor.defaultFormatter": "charliermarsh.ruff",

        // 저장할 때 코드 포맷팅
        "editor.formatOnSave": true,

        // 저장할 때 Ruff 자동 수정 및 Import 정리
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },

    // 개인적으로 선호하는 Ruff 포맷 설정
    "ruff.configuration": {
        "format": {
            // trailing comma가 있어도 그것만을 이유로 줄바꿈을 유지하지 않음
            "skip-magic-trailing-comma": true
        }
    },

    // 프로젝트의 pyproject.toml 설정을 개인 설정보다 우선
    "ruff.configurationPreference": "filesystemFirst"
}
```

### ③ Ruff 재시작

1. `Ctrl + Shift + P`를 누릅니다.
2. `Ruff: Restart Server`를 실행합니다.

이제 Python 파일을 저장하면 Ruff가 코드 포맷팅, 자동 수정, Import 정리를 실행합니다.

## 3. 프로젝트에서는 pyproject.toml 사용

`settings.json`은 내 컴퓨터에서 사용하는 개인 설정이고, `pyproject.toml`은 프로젝트에 포함해 팀원들과 공유하는 설정 파일입니다.

팀 프로젝트에서는 프로젝트 루트에 `pyproject.toml`을 만들고, 팀에서 합의한 Ruff 규칙을 작성합니다.

예시:

```toml
[tool.ruff]
line-length = 88

[tool.ruff.format]
# 팀에서 정한 포맷 규칙
skip-magic-trailing-comma = false
```

위 예시에서는 프로젝트가 `skip-magic-trailing-comma = false`를 지정했으므로, 개인 설정의 `true`보다 프로젝트 설정인 `false`가 우선합니다. `pyproject.toml`가 우선이 되는 예시를 보인 것뿐이며, `skip-magic-trailing-comma = false`를 권장하는 것은 아닙니다.

정리하면 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| VS Code `settings.json` | 저장할 때 Ruff를 실행하고, 개인적으로 선호하는 기본 설정을 지정 |
| 프로젝트 `pyproject.toml` | 해당 프로젝트에서 팀원 모두가 따를 Ruff 규칙을 지정 |

**개인 설정은 기본값으로 사용하고, 팀 프로젝트에 별도 규칙이 있으면 프로젝트 설정을 따릅니다.**

---

참고: [Ruff 공식 VS Code 설정 문서](https://docs.astral.sh/ruff/editors/settings/) · [Ruff 포맷 설정 문서](https://docs.astral.sh/ruff/settings/)
