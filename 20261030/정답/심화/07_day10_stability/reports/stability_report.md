# 강사용 보고서 예시 · 실제 CPU 실행 결과

|후보|best epoch|stop epoch|best valid loss|last valid loss|
|---|---:|---:|---:|---:|
|baseline|40|43|0.020061|0.022444|
|dropout|28|36|0.035909|0.044228|
|batchnorm|21|29|0.035851|0.051663|
|weight_decay|46|51|0.013377|0.015681|

선택: weight_decay. 같은 4후보 목록에서 best valid loss가 가장 작은 후보이며 test는 보지 않았다.
모든 후보에 같은 Plateau·Early Stopping 정책을 적용했다. 실제 stop epoch가 다르므로 동일 계산량 실험이라고 말하지 않는다.
관측 최저 모델을 복원한 loss는 기록된 best loss와 같았다. min_delta 기준의 patience는 별도 reference에 적용했다.
이번 실행에서 규제를 넣은 모든 후보가 기준선보다 좋아지지는 않았다. 이름이 아니라 실제 valid 근거를 쓴다.

## 오류 처방의 예시 (아래는 별도 가정 사례)
- train/valid가 함께 정체: 학습률·gradient 계산·입력 스케일을 확인하고 하나만 바꾼다.
- train만 좋아지고 valid가 악화: best epoch와 데이터 분포를 확인한 후 규제 하나를 비교한다.
- NaN: epoch를 늘리기보다 입력·loss 사용·학습률·gradient norm을 점검한다.
- 작은 배치의 BN 변동: 배치별 통계와 마지막 배치 크기를 먼저 확인한다.

한계: 작은 합성 자료, seed 한 번, valid 반복 선택의 낙관 편향. 독립 test 결과는 아직 없다.
