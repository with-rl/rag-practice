import os
import argparse
import re
import json


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--wiki_home", type=str, default="data/kowiki")
    p.add_argument("--dump_file", type=str, default="data/kowiki_dump.txt")

    p.add_argument("--n_file", type=int, default=-1)

    args = p.parse_args()

    return args


def main(args):
    files = []
    for name in os.listdir(args.wiki_home):
        path = os.path.join(args.wiki_home, name)
        if os.path.isdir(path):
            for name in os.listdir(path):
                if re.match(r"wiki_[0-9]{2}", name):
                    files.append(os.path.join(path, name))

    with open(args.dump_file, "w") as f_out:
        for i, file in enumerate(sorted(files)):
            if args.n_file > 0 and i >= args.n_file:
                break
            with open(file) as f_in:
                for line in f_in:
                    text = json.loads(line)["text"]
                    f_out.write(text)
                    f_out.write("\n" * 4)


if __name__ == "__main__":
    args = get_args()
    main(args)
