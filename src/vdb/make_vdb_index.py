import argparse
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dump_file", type=str, default="data/kowiki_dump.txt")
    p.add_argument("--vdb_index", type=str, default="data/kowiki_vdb")
    p.add_argument(
        "--context_embedding_id",
        type=str,
        default="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    )

    args = p.parse_args()

    return args


def main(args):
    # document load
    loader = TextLoader(args.dump_file)
    documents = loader.load()
    # document split
    doc_splitter = CharacterTextSplitter(
        separator="\n\n\n\n", chunk_size=200000, chunk_overlap=0
    )
    docs = doc_splitter.split_documents(documents)
    # text split
    text_splitter = CharacterTextSplitter(
        separator="", chunk_size=500, chunk_overlap=100, length_function=len
    )
    # make faiss index
    embed_model = HuggingFaceEmbeddings(
        model_name=args.context_embedding_id,
        # model_kwargs={"device": "cuda"},
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = None

    for doc in tqdm(docs):
        texts = text_splitter.split_text(doc.page_content)
        if len(texts) > 0:
            if db is None:
                db = FAISS.from_texts(texts, embed_model)
            else:
                db.add_texts(texts)
    # save faiss index
    db.save_local(args.vdb_index)


if __name__ == "__main__":
    args = get_args()
    main(args)
