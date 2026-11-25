"""저장된 Day28 실습 07 평가 결과로 Markdown 보고서를 생성합니다."""

import argparse
import json
from pathlib import Path


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


def format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def render_report(result: dict) -> str:
    """실행 정보, 질문별 진단, 변형별 검색 지표를 Markdown으로 반환합니다."""
    lines = [
        "# Day28 실습 07 평가 보고서",
        "",
        f"- 실행 모드 `{result['mode']}`",
        f"- 데이터셋 버전 `{result['dataset_version']}`",
        "",
        "## 질문별 진단",
        "",
        "| 질문 | Baseline Recall | Rerank Recall | Recall 변화량 | 진단 | 배포 보류 | 판단 근거 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    comparisons = result["comparisons"]
    for row in comparisons:
        decision = row["regression"]
        delta = row["delta_recall"]
        delta_text = "N/A" if delta is None else f"{delta:+.3f}"
        hold = "예" if decision["hold_release"] else "아니요"
        lines.append(
            f"| {row['question_id']} | {format_score(row['baseline_recall'])} | "
            f"{format_score(row['rerank_recall'])} | {delta_text} | "
            f"`{row['diagnosis']}` | {hold} | {decision['reason']} |"
        )

    lines.extend([
        "",
        "## 실험 변형별 검색 지표",
        "",
        "| 질문 | 실험 변형 | k | Gold 수 | 검색 수 | Recall@k | Precision@k | Hit@k | RR@k | DCG@k | nDCG@k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        for variant, metrics in row["matrix"].items():
            hit = metrics["hit_at_k"]
            hit_text = "N/A" if hit is None else str(hit)
            lines.append(
                f"| {row['question_id']} | `{variant}` | {metrics['k']} | "
                f"{metrics['gold_count']} | {metrics['retrieved_count']} | "
                f"{format_score(metrics['recall_at_k'])} | {format_score(metrics['precision_at_k'])} | "
                f"{hit_text} | {format_score(metrics['rr_at_k'])} | "
                f"{format_score(metrics['dcg_at_k'])} | {format_score(metrics['ndcg_at_k'])} |"
            )

    lines.extend([
        "",
        "`N/A`는 Gold가 없는 No-answer 질문에서 검색 지표가 산출되지 않았음을 뜻합니다.",
        "배포 보류 여부는 실습 기준에 따른 판단이며, 최종 배포에는 별도 검토가 필요합니다.",
    ])
    if result["mode"] == "offline_fixture":
        lines.append("이번 결과는 교육용 고정 데이터로 수행한 버전 비교입니다.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUTS / "result.json",
                        help="평가 결과 JSON 경로")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "report.md",
                        help="Markdown 보고서 저장 경로")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as source:
        result = json.load(source)
    report = render_report(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
