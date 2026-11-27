"""Day27 강의 실습. 실행은 역할 폴더에서 Python으로 실행합니다."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())
sys.path.insert(0,str(ROLE_ROOT))


import argparse
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from coursekit.workshops import STAGE_RUNNERS
from coursekit.io import save_report
from coursekit.rewrite import RewriteResult


def prepare_pipeline_request(case):
    """원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다."""

    # TODO D27-06-B01: 원 질문과 History를 함께 보존하고 검색용 질문은 Chain 내부에서 만듭니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.
    return {
        "origin_question": case["question"],
        "history": case["history"],
        "category": case["category"],
        "k": 5,
        "role": "learner",
    }


def wire_langchain_pipeline(dependencies, offline):
    rewrite_fn, search_fn, rerank_fn, generation_fn = dependencies
    # dependencies: rewrite, search, rerank, generation
    # rewrite: original_question, history -> RewriteResult
    # search: search_query -> Document
    # rerank: q,docs,k -> Document
    # generation: question, context -> dict(WireAnswer 포함)
    # 여기서 반환하는 chain을 invoke 호출할 때 prepare_pipeline_request 가 반환하는 dict 전달 
    """LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다."""
    # TODO D27-06-B02: LCEL는 Runnable을 준비할 뿐이고 .invoke()가 실행 시작점입니다.
    # 반환값과 예외 조건은 README.md의 수행 기준을 따르세요.
    '''
        assign은 들어온 dict를 그대로 두고 새 키만 덧붙여서 다음 단계로 넘깁니다.
        그래서 rerank 단계에서도 x["category"]나 x["rewrite_result"]를 그대로 꺼낼 수 있습니다.

        다만 누적은 assign일 때만 일어납니다.
        마지막 generation_step처럼 일반 RunnableLambda를 쓰면 그 함수의 반환값만 다음으로 넘어가고, 그때까지 쌓인 dict는 사라집니다. 그래서 체인의 최종 결과는 WireAnswer 하나가 됩니다.
        중간 값도 결과에 남기고 싶다면 마지막 단계도 assign으로 바꾸면 됩니다.

        반대로 일부 키만 남기고 싶다면 .pick()을 씁니다.

        누적된다고 해서 데이터가 복사되는 것은 아닙니다.
        dict에는 같은 객체를 가리키는 참조만 담기므로, 문서가 많아도 메모리 부담은 거의 없습니다.
        
        최종 반환 형태는 README에 정해진 대로(WireAnswer 하나인지 dict인지) 맞추면 됩니다.
    '''
    print('rewrite_fn: ', rewrite_fn)
    rewrite_step = RunnablePassthrough.assign(
        rewrite_result = RunnableLambda(
            lambda x: rewrite_fn(
                q=x["origin_question"],
                h=x["history"]
            )
        )
    )
    
    search_step = RunnablePassthrough.assign(
        documents = RunnableLambda(
            lambda x: search_fn(
                question=x["rewrite_result"].search_query,
                category=x["category"],
                k=x["k"],
                role=x["role"],
            )
        )
    )

    rerank_step = RunnablePassthrough.assign(
        reranked = RunnableLambda(
            lambda x: rerank_fn(
                q=x["rewrite_result"].search_query,
                docs=x["documents"],
                k=x["k"],
            )
        )
    )

    #generation_step = RunnableLambda(
    generation_step = RunnablePassthrough.assign(
        generation = RunnableLambda(
            lambda x: generation_fn(
                q=x["rewrite_result"].search_query,
                ctx=x["reranked"]
            )
        )
    )

    # LCEL로 Runnable 준비
    if not offline:
        return rewrite_step | search_step | rerank_step | generation_step
    else:
        pass


def main():
    parser=argparse.ArgumentParser(description='Day27 실습 06 · 06_langchain_pipeline')
    parser.add_argument('--offline',action='store_true',help='교육용 고정 데이터·점수. 모델 품질 실측이 아닙니다.')
    args=parser.parse_args()
    # Day27: 일부 진단 실습 외에는 기본 실행이 실제 모델·DB·LLM 경로입니다.
    report=STAGE_RUNNERS[6](args.offline,[prepare_pipeline_request,wire_langchain_pipeline])
    path=save_report(HERE,'day27_06_'+('fixture' if args.offline else 'live')+'.json',report)
    print('실습 06 결과:', path)
    print('보고서를 확인하고 reports/review.md에 원문 대조·실행 모드를 기록하세요.')

if __name__=='__main__':main()




'''
    [Q] build_rewrite 에서 받는 입력을 build_rerank 에서 접근하려면?

    build_rewrite()에 들어가는 원본 입력을 RunnablePassthrough()로 보관하고, 검색 단계의 반환값에도 담아 넘기면 build_rerank()에서 꺼낼 수 있어요.

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

def build_search():
    def search(data):
        query = data["rewrite_result"].search_query
        return {
            "original_input": data["original_input"],
            "query": query,
            "documents": search_fn(query),
        }

    return RunnableLambda(search)

def build_rerank(k):
    return RunnableLambda(
        lambda data: rerank_fn(
            q=data["original_input"]["question"],
            docs=data["documents"],
            k=k,
        )
    )

chain = (
    RunnableParallel(
        original_input=RunnablePassthrough(),
        rewrite_result=build_rewrite(),
    )
    | build_search()
    | build_rerank(k=5)
)

result = chain.invoke({"question": question})

.invoke({"question": question})의 딕셔너리가 두 곳에 전달됩니다. original_input에는 그 딕셔너리가 그대로 들어가고, rewrite_result에는 재작성 결과가 들어갑니다. 검색 단계가 original_input을 다시 반환하므로 재정렬 단계에서도 data["original_input"]["question"]으로 원본 질문을 읽을 수 있어요.

체인에 질문 문자열을 직접 넘기는 형태라면 q=data["original_input"]으로 쓰면 됩니다.
'''