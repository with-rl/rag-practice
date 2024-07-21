import os
import argparse
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--context", type=str, default="data/aihub/eval_context.json")
    p.add_argument("--question", type=str, default="data/aihub/eval_question.json")
    p.add_argument(
        "--model_id", type=str, default="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    )

    args = p.parse_args()
    return args


def main(args):
    with open(args.context) as f:
        context = json.load(f)

    with open(args.question) as f:
        question = json.load(f)

    assert len(context) == len(question)

    # SentenceBERT 모델 생성
    model = SentenceTransformer(args.model_id)

    # corpus embeddings
    context_keys = np.array(list(context.keys()))
    context_values = [context[k] for k in context_keys]
    context_embeddings = model.encode(context_values, normalize_embeddings=True)

    # 평가
    score = 0.0
    for key, value in tqdm(question.items()):
        # query embedding
        question_embedding = model.encode(value["question"], normalize_embeddings=True)
        # score 계산
        scores = np.dot(context_embeddings, question_embedding)
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
