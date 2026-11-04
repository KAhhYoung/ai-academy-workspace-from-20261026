
import torch
from torch import nn
from pathlib import Path
import pandas as pd

torch.set_num_threads(1)
torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU(CUDA)를 사용할 수 있으면 GPU, 아니면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

if device.type == "cuda":
    print("GPU 이름:", torch.cuda.get_device_name(0))

# 입력 데이터를 GPU로 이동
x = torch.randn(2, 5, 16, device=device)
lengths = torch.tensor([3, 5], device=device)

# Padding Mask 생성 (GPU)
# 첫 번째 샘플: 앞의 3개 토큰만 유효
# 두 번째 샘플: 5개 토큰 모두 유효
mask = torch.arange(5, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

# Padding 위치의 입력값을 0으로 설정
x[mask] = 0

# Multi-Head Attention 모델을 GPU로 이동
attention = nn.MultiheadAttention(
    embed_dim=16,
    num_heads=4,
    dropout=0,
    batch_first=True
).to(device).eval()

# 추론 모드: Gradient 계산 비활성화
with torch.inference_mode():

    # Padding Mask를 사용하지 않은 Attention
    plain, _ = attention(x, x, x)

    # Padding Mask를 적용한 Self-Attention
    # key_padding_mask: True인 위치를 Attention 대상에서 제외
    # average_attn_weights=False: Head별 Attention 가중치를 개별 반환
    masked, weights = attention(
        x, x, x,
        key_padding_mask=mask,
        average_attn_weights=False
    )

    # Key Mask만으로는 Padding Query의 출력까지 0이 되지 않으므로
    # 평균 계산 시 Padding 위치를 제외
    mean = (
        (masked * (~mask).unsqueeze(-1)).sum(1)
        / lengths.unsqueeze(-1)
    )

print("shape:", tuple(masked.shape), tuple(weights.shape))
print(
    "막은 key weight 최대:",
    round(weights[0, :, :, 3:].abs().max().item(), 6)
)
print(
    "유효 query 출력 변화:",
    round((plain[0, :3] - masked[0, :3]).abs().max().item(), 6)
)
print("masked mean:", tuple(mean.shape))

# GPU Tensor는 NumPy로 직접 변환할 수 없으므로
# .cpu()로 CPU 메모리로 이동한 뒤 NumPy로 변환
plt.figure(figsize=(6, 3))
plt.imshow(mask.int().cpu().numpy(), vmin=0, vmax=1)
plt.xlabel("Time")
plt.ylabel("Sample")
plt.title("Padding mask: 1 = ignored")
plt.xticks(range(5))
plt.yticks(range(2))
plt.tight_layout()
plt.savefig(OUT / "mask_or_latent.png", dpi=120)
plt.close()
