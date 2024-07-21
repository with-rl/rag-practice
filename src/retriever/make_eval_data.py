import os
import glob
import argparse
import json
from zipfile import ZipFile
import numpy as np


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--ai_hub_home", type=str, default="data/aihub")

    args = p.parse_args()
    return args


def parse_paragraphs(code, files, count):
    # eval count per file
    n_eval = count // len(files) + (0 if count % len(files) == 0 else 1)
    # return dicts
    question, context = dict(), dict()
    # sample from json
    for i, file in enumerate(files):
        with ZipFile(file) as z:
            assert len(z.namelist()) == 1
            with z.open(z.namelist()[0]) as f:
                data = json.load(f)["data"]
        paragraphs = []
        for row in data:
            paragraphs.extend(row["paragraphs"])
        assert len(paragraphs) > n_eval
        # random sample
        for idx in sorted(
            np.random.choice(range(len(paragraphs)), n_eval, replace=False)
        ):
            # unique key
            key = f"{code}-{i:02d}-{idx:06d}"
            assert key not in context and key not in question
            # paragraph
            paragraph = paragraphs[idx]
            # context
            context[key] = paragraph["context"]
            # question
            qas = paragraph["qas"]
            assert len(qas) > 0
            question[key] = {"question": qas[0]["question"], "positive": key}
    # return
    return question, context


def parse_passage(code, files, count):
    # eval count per file
    n_eval = count // len(files) + (0 if count % len(files) == 0 else 1)
    # return dicts
    question, context = dict(), dict()
    # sample from json
    for i, file in enumerate(files):
        with ZipFile(file) as z:
            assert len(z.namelist()) == 1
            with z.open(z.namelist()[0]) as f:
                data = json.load(f)["data"]
        paragraphs = data
        assert len(paragraphs) > n_eval
        # random sample
        for idx in sorted(
            np.random.choice(range(len(paragraphs)), n_eval, replace=False)
        ):
            # unique key
            key = f"{code}-{i:02d}-{idx:06d}"
            assert key not in context and key not in question
            # paragraph
            paragraph = paragraphs[idx]
            # context
            context[key] = paragraph["passage"]
            # question
            qas = paragraph["qa_pairs"]
            assert len(qas) > 0
            question[key] = {"question": qas[0]["question"], "positive": key}
    # return
    return question, context


def main(args):
    question, context = dict(), dict()
    np.random.seed(1234)
    item_count = 2047

    # 016.행정 문서 대상 기계독해 데이터
    files = glob.glob(
        os.path.join(
            args.ai_hub_home,
            "016.행정 문서 대상 기계독해 데이터/01.데이터/2.Validation/라벨링데이터",
            "*.zip",
        )
    )
    q, c = parse_paragraphs(
        "106",
        files,
        item_count,
    )
    question.update(q)
    context.update(c)

    # 017.뉴스 기사 기계독해 데이터
    files = glob.glob(
        os.path.join(
            args.ai_hub_home,
            "017.뉴스 기사 기계독해 데이터/01.데이터/2.Validation/라벨링데이터",
            "*.zip",
        )
    )
    q, c = parse_paragraphs(
        "107",
        files,
        item_count,
    )
    question.update(q)
    context.update(c)

    # 021.도서자료 기계독해
    files = glob.glob(
        os.path.join(
            args.ai_hub_home,
            "021.도서자료 기계독해/01.데이터/2.Validation/라벨링데이터",
            "*/*.zip",
        )
    )
    q, c = parse_paragraphs(
        "021",
        files,
        item_count,
    )
    question.update(q)
    context.update(c)

    # 150.숫자연산 기계독해 데이터
    files = glob.glob(
        os.path.join(
            args.ai_hub_home,
            "150.숫자연산 기계독해 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터",
            "*.zip",
        )
    )
    q, c = parse_passage(
        "150",
        files,
        item_count,
    )
    question.update(q)
    context.update(c)

    # 151.금융, 법률 문서 기계독해 데이터
    files = glob.glob(
        os.path.join(
            args.ai_hub_home,
            "151.금융, 법률 문서 기계독해 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터",
            "*.zip",
        )
    )
    q, c = parse_paragraphs(
        "151",
        files,
        item_count,
    )
    question.update(q)
    context.update(c)

    # # 152.기술과학 문서 기계독해 데이터
    # files = glob.glob(
    #     os.path.join(
    #         args.ai_hub_home,
    #         "152.기술과학 문서 기계독해 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터",
    #         "*.zip",
    #     )
    # )

    # # 기계독해/기계독해분야
    # files = glob.glob(
    #     os.path.join(
    #         args.ai_hub_home,
    #         "기계독해/기계독해분야",
    #         "*.zip",
    #     )
    # )

    # check duplicate question & context
    set_context = set()
    for key in list(context.keys()):
        text = context[key]
        if text in set_context:
            print(f"duplicate: {key} : {text}")
            del context[key]
            del question[key]
        else:
            set_context.add(text)

    set_question = set()
    for key in list(question.keys()):
        text = question[key]["question"]
        if text in set_question:
            print(f"duplicate: {key} : {text}")
            del context[key]
            del question[key]
        else:
            set_question.add(text)

    # save file
    print(f"question={len(question)}, context={len(context)}")
    with open(os.path.join(args.ai_hub_home, "eval_question.json"), "w") as f:
        json.dump(question, f, indent=" ", ensure_ascii=False)
    with open(os.path.join(args.ai_hub_home, "eval_context.json"), "w") as f:
        json.dump(context, f, indent=" ", ensure_ascii=False)


if __name__ == "__main__":
    args = get_args()
    main(args)
