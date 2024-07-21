import os
import argparse
import json
import numpy as np

from mecab import MeCab
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--context", type=str, default="data/aihub/eval_context.json")
    p.add_argument("--question", type=str, default="data/aihub/eval_question.json")

    args = p.parse_args()
    return args


def main(args):
    with open(args.context) as f:
        context = json.load(f)

    with open(args.question) as f:
        question = json.load(f)

    assert len(context) == len(question)

    # 형태소 분석기를 이용한 tokeinizer 선언
    # 조사 등 일부 품사를 제거
    # 품사표: https://blog.naver.com/aramjo/221404488280
    MECAB = MeCab()
    EXCLUDE = set(
        [
            "JKS",
            "JKC",
            "JKG",
            "JKO",
            "JKB",
            "JKV",
            "JKQ",
            "JX",
            "JC",
            "EP",
            "EF",
            "EC",
            "ETN",
            "ETM",
            "SF",
            "SSC",
            "SSO",
            "SY",
        ]
    )

    def tokenizer(sent):
        tokens = []
        for w, t in MECAB.pos(sent):
            if t not in EXCLUDE:
                tokens.append(w)
        return tokens

    # tokenize
    context_keys = np.array(list(context.keys()))
    tokenized_contexts = [tokenizer(context[k]) for k in context_keys]

    # bm25 class 생성
    bm25 = BM25Okapi(tokenized_contexts)

    # 평가
    score = 0.0
    for key, value in tqdm(question.items()):
        # question tokenize
        tokenized_question = tokenizer(value["question"])
        # score 계산
        scores = bm25.get_scores(tokenized_question)
        # score 역순으로 정렬
        rank = np.argsort(-scores)[:10]  # top 10
        # mrr 계산
        rank_keys = context_keys[rank]
        result = np.where(rank_keys == key)
        assert len(result[0]) < 2
        if len(result[0]) == 1:
            score += 1 / (result[0][0] + 1)
    print(score / len(question))


if __name__ == "__main__":
    args = get_args()
    main(args)
