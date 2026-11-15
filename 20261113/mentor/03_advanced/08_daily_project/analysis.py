from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LEVEL_ROOT = HERE.parent
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
import math
import torch
import torch.nn.functional as F
from common.support import DATA, ToyTokenizer, attention, cosine, write_json, write_md

torch.manual_seed(42)
torch.set_num_threads(1)


'''
DATA = [
    {"id":"A", "title":"실습 파일 보관", "content":"실습 파일은 개인 폴더에 보관합니다.", "category":"learning"},
    {"id":"B", "title":"서버 실행", "content":"프로젝트 폴더에서 서버를 실행합니다.", "category":"environment"},
    {"id":"C", "title":"질문 작성", "content":"오류 메시지와 시도한 작업을 함께 적습니다.", "category":"learning"},
]
'''
# PPT 78~85장의 일일프로젝트. 앞선 계산을 재사용하는 통합 결과입니다.
# 이 코드는 실제 학습된 LLM이 아니며 예시 Token·Vector·Attention·Logit을 결합합니다.
def run_analysis():
    texts=[d["content"] for d in DATA]
    tokenizer=ToyTokenizer(texts)

    # --------------------------------
    # 1. 입력 시퀀스 준비
    # input_sentence="실습 파일은 개인 폴더에 보관합니다."
    # --------------------------------
    input_sentence=DATA[0]["content"]

    # --------------------------------
    # 2. toy 토크나이저로 인코딩
    # 토크나이저로 인코딩한 결과는 토큰 ID(정수). 이 ID를 임베딩 층에 넣으면 임베딩 벡터가 됨
    # --------------------------------
    token_report=tokenizer.encode(input_sentence,max_length=12)
    assert token_report["valid_length"]<=12
    # 해설: 실제 모델의 Tokenizer가 아닌 규칙 기반 교육용 토크나이저입니다.

    # --------------------------------
    # 3. A좌표와의 코사인 유사도 구하기 (A_B, A_C)
    # TODO: 코사인 유사도 어디에서 사용되는지 확인
    # --------------------------------
    coords={"A":[1.,0.],"B":[.8,.2],"C":[-1.,0.]}
    cosine_table={f"A_{k}":cosine(coords["A"],v) for k,v in coords.items() if k!="A"}
    # 해설: 문장 B가 이웃처럼 보이는 것은 교육용 좌표를 그렇게 부여했기 때문입니다.

    # --------------------------------
    # 4. self-attention: attention scores,weights,가중합 구하기 | **causal mask 사용**
    # --------------------------------
    q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    scores,weights,attn_out,mask=attention(q,q,q,causal=True)

    # weights에 softmax 적용여부 검증
    # weights.sum(dim=-1): PyTorch의 sum()은 기본적으로 합산한 차원을 제거하므로 (3,)이 됨 
    assert torch.allclose(weights.sum(dim=-1),torch.ones(3))

    # scores에 causal mask 적용여부 검증
    assert torch.all(weights[mask]==0)
    # 해설: 이 Attention은 실제 지식 문장의 자연어 의미를 이해한 결과가 아닙니다.


    # --------------------------------
    # 5. 다음 토큰 선택 원리
    # --------------------------------
    logits=torch.tensor([2.,1.,0.])
    probs=torch.softmax(logits,dim=0)
    selected=int(probs.argmax().item())
    next_token=["출력A","출력B","출력C"][selected]
    # 해설: 이 예시는 후보 선택 원리만 보여 주고 실제 자연어 생성은 하지 않습니다.



    # --------------------------------
    # 위의 항목별 원리를 작은 예시에 적용
    # --------------------------------




    # --------------------------------
    # 1. 입력 시퀀스 준비
    # --------------------------------
    custom_texts=["실습 기록은 개인 폴더에 보관합니다.", "서버를 프로젝트 폴더에서 실행합니다.",
                  "오류 메시지와 실행 명령을 같이 적습니다."]
    # 교육용 어휘표를 새 텍스트로 고정하고 길이 차이를 비교합니다.


    # --------------------------------
    # 2. toy 토크나이저로 인코딩 및 padding mask 생성... TODO: 그런데 적용은 안함..?
    # --------------------------------
    custom_tokenizer=ToyTokenizer(custom_texts)
    custom_batch=custom_tokenizer.batch(texts=custom_texts,max_length=12)
    # 시퀀스별 토큰길이
    custom_lengths=[row["valid_length"] for row in custom_batch]

    
    # --------------------------------
    # 3. self-attention: 가중치 구하기 | **causal mask 적용 안함**
    # --------------------------------
    _,plain_w,_,_=attention(q,q,q,causal=False)
    mask_diff=float(torch.abs(weights-plain_w).sum())
    # END D20-08-A06

    platform_map={
        "FastAPI":"HTTP 입력·검증·응답; 지식 관리 엔드포인트",
        "PostgreSQL":"지식 원본·분류·업데이트 기록(후속 설계)",
        "LLM":"입력 Token에 조건부로 다음 Token 분포 생성",
        "Retriever_RAG":"관련 근거 검색·출처 제공(11/20 이후 구현)",
        "Agent":"도구 선택·단계 실행, 권한 확인 필요(후속 구현)",
    }

    limitations=[
        "Tokenizer는 실제 사전학습 모델의 서브워드 Tokenizer가 아니다",
        "벡터는 임의 교육용 좌표로 문장 의미 검색 성능이 없다",
        "Attention은 자연어 모델의 학습된 Head가 아니라 장난감 Score 예제이다",
        "Logit은 예시 세 수치이며 실제 모델 vocabulary 분포가 아니다",
        "RAG·Agent 동작을 구현한 것이 아니며 이후 과목에서 연결한다",
    ]

    lines=[
        "# 일일프로젝트: 문장 하나가 LLM 출력이 되기까지", "",
        f"입력: {input_sentence}", "",
        f"- Token 및 Special Token 포함 입력 길이: {token_report['valid_length']}",
        f"- Token IDs: {token_report['input_ids']}",
        f"- 교육용 Cosine A-B / A-C: {cosine_table['A_B']:.4f} / {cosine_table['A_C']:.4f}",
        f"- Causal Mask Attention Row Sum: {weights.sum(dim=-1).tolist()}",
        f"- 다음 Token 예시 선택: {next_token}", "",
        "```mermaid", "flowchart LR", "  U[입력 문장] --> T[Token/ID] --> E[Embedding] --> A[Attention/Mask] --> L[Logit/Softmax] --> O[다음 Token]",
        "  R[후속 Retriever] -. 검색 근거 .-> T", "  DB[(원본 PostgreSQL)] -. 원본/메타데이터 .-> R",
        "  F[FastAPI] -. 입력/응답 .-> U", "```", "",
        "주의: 그림은 개념적 위치 표시이며 오늘 실제 RAG·Agent 서비스를 실행한 것이 아닙니다.",
    ]
    write_md(HERE/"outputs"/"one_page_summary.md","\n".join(lines)+"\n")

    # BEGIN D20-08-A10
    write_json(HERE/"outputs"/"daily_project.json",{
        "source_data":DATA,"token":token_report,"cosine":cosine_table,
        "scores_shape":list(scores.shape),"attention_weights":weights.tolist(),
        "row_sums":weights.sum(dim=-1).tolist(),"causal_mask":mask.tolist(),
        "logits":logits.tolist(),"probabilities":probs.tolist(),
        "next_token_demo":next_token,"custom_token_lengths":custom_lengths,
        "masked_unmasked_l1":mask_diff,"platform":platform_map,
        "limitations":limitations,"scope":"교육용 독립 수치 계산; 실제 LLM/RAG/Agent 아님",
    })
    print("[08] daily_project.json + one_page_summary.md 생성 완료")
    # END D20-08-A10

