from pymilvus import MilvusClient

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)

from langserve import CustomUserType


PROMPT_TEMPLATE = """당신이 가진 지식보다 아래 내용을 내용을 참고해서 '질문'에 대해서 답변해 주세요.:

{context}

질문: {question}
"""


class Question(CustomUserType):
    question: str


class FAISSChain:
    def __init__(self, args):
        # loading vdb index
        embed_model = HuggingFaceEmbeddings(
            model_name=args.question_embedding_id,
            # model_kwargs={"device": "cuda"},
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        db = FAISS.load_local(
            args.vdb_index, embed_model, allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": args.top_k})

        # llm
        llm = HuggingFaceEndpoint(
            repo_id=args.llm_model_id,
            max_new_tokens=1024,
            temperature=0.1,
        )

        # prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        self.chain = (
            RunnableLambda(self.custom_input)
            | {
                "context": retriever | self.format_contexts,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

    def custom_input(self, question: Question) -> str:
        assert isinstance(question, Question)
        return question.question

    # contxt formatt
    def format_contexts(self, contexts):
        return "\n\n".join(
            [
                d.page_content.replace("&lt;", "<").replace("&gt;", ">").strip()
                for d in reversed(contexts)
            ]
        )


class MilvusChain:
    def __init__(self, args):
        # loading vdb index
        self.embed_model = HuggingFaceEmbeddings(
            model_name=args.question_embedding_id,
            # model_kwargs={"device": "cuda"},
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.client = MilvusClient(args.milvus_vdb_index)

        # llm
        llm = HuggingFaceEndpoint(
            repo_id=args.llm_model_id,
            max_new_tokens=1024,
            temperature=0.1,
        )

        # prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        self.chain = (
            RunnableLambda(self.custom_input)
            | {
                "context": self.milvus_retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

    def custom_input(self, question: Question) -> str:
        assert isinstance(question, Question)
        return question.question

    def milvus_retriever(self, question: str) -> str:
        assert isinstance(question, str)
        query_vectors = self.embed_model.embed_query(question)
        res = self.client.search(
            collection_name="kowiki_collection",
            data=[query_vectors],
            # filter="subject == 'history'",
            limit=5,
            output_fields=["text", "subject"],
        )
        context = []
        for r in res[0]:
            context.append(
                r["entity"]["text"].replace("&lt;", "<").replace("&gt;", ">").strip()
            )
        return "\n\n".join(context)
