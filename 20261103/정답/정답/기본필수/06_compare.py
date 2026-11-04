
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(42)

# ==========================================
# 1. GPU 사용 환경 설정
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

if device.type == "cuda":
    print("GPU 이름:", torch.cuda.get_device_name(0))

# ==========================================
# 2. 데이터 생성
# ==========================================
# 원리용 독립 Sequence 20개
# 증가=1, 감소=0, 가운데 평균은 거의 같다.

g = torch.Generator().manual_seed(1203)
y_all = torch.tensor([0, 1] * 10)
base = torch.linspace(-1, 1, 6)

x_all = torch.stack(
    [(2 * y - 1) * base for y in y_all]
).unsqueeze(-1)

x_all = x_all + 0.03 * torch.randn(20, 6, 1, generator=g)

# 학습/검증 데이터 분리 후 GPU로 이동
x_train = x_all[:12].to(device)
y_train = y_all[:12].to(device)

x_valid = x_all[12:16].to(device)
y_valid = y_all[12:16].to(device)

# 마지막 4개는 보관. 함수에 전달하지 않는다.

# ==========================================
# 3. RNN / LSTM 학습 및 검증
# ==========================================
for kind in ["RNN", "LSTM"]:
    torch.manual_seed(42)

    # RNN 또는 LSTM 생성 후 GPU로 이동
    core = (
        nn.RNN(1, 8, batch_first=True)
        if kind == "RNN"
        else nn.LSTM(1, 8, batch_first=True)
    ).to(device)

    # 분류용 Linear Head도 GPU로 이동
    head = nn.Linear(8, 2).to(device)

    parameters = list(core.parameters()) + list(head.parameters())

    optimizer = torch.optim.Adam(parameters, lr=0.02)

    # 학습 모드 설정
    core.train()
    head.train()

    for epoch in range(30):
        optimizer.zero_grad()

        # GPU에서 RNN/LSTM 순전파 수행
        output, _ = core(x_train)

        # 마지막 Time Step의 출력을 사용하여 분류
        loss = nn.CrossEntropyLoss()(
            head(output[:, -1]),
            y_train
        )

        # 손실의 Gradient를 계산하고 가중치 갱신
        # GPU에서 역전파 및 파라미터 업데이트 수행
        loss.backward()
        optimizer.step()

    # ==========================================
    # 4. 검증
    # ==========================================
    core.eval()
    head.eval()

    with torch.inference_mode():
        # GPU에서 검증 데이터 추론
        output, _ = core(x_valid)
        pred = head(output[:, -1]).argmax(dim=1)

    print(
        kind,
        "params:", sum(p.numel() for p in parameters),
        "valid 정답:", y_valid.cpu().tolist(),
        "예측:", pred.cpu().tolist(),
        "accuracy:", round(
            (pred == y_valid).float().mean().item(), 3
        )
    )
