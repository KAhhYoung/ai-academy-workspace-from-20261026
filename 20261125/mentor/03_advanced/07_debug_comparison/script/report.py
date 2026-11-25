"""챕터 7 실습 결과를 질문·실험 변형별 JSONL로 저장합니다."""

import json
from pathlib import Path


'''
Day28 강의 자료의 챕터 7과 기존 main()을 바탕으로 ChatGPT에
평가 결과를 JSONL로 저장하는 script/report.py 작성을 요청했다.

main()은 --offline 또는 --live 옵션에 따라 평가를 실행하고,
전체 결과를 outputs/result.json에 저장한다. 이어서 report.py의
write_eval_runs()를 호출해 outputs/eval_runs.jsonl을 생성한다.

write_eval_runs()는 질문과 실험 변형의 조합마다 평가 결과를
JSONL 한 줄로 저장한다. main.py를 한 번 실행하면 두 파일이 모두 생성된다.

저장된 outputs/result.json으로 JSONL만 다시 만들려면
실습 폴더에서 python script/report.py를 실행한다.
'''

def write_eval_runs(result: dict, output: Path) -> Path:
    """이미 계산된 비교 지표를 한 줄에 하나씩 기록합니다."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report:
        for comparison in result["comparisons"]:
            for variant, metrics in comparison["matrix"].items():
                row = {
                    "question_id": comparison["question_id"],
                    "variant": variant,
                    "recall_at_3": metrics["recall_at_k"],
                    "mode": result["mode"],
                    "dataset_version": result["dataset_version"],
                }
                report.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    with (output_dir / "result.json").open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    print(write_eval_runs(result, output_dir / "eval_runs.jsonl"))
