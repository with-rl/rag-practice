from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_core.beta.runnables.context import Context


PROMPT_TEMPLATE = """당신이 가진 지식보다 아래 내용을 내용을 참고해서 '질문'에 대해서 답변해 주세요.:

{context}

질문: {question}
"""


# LangChain 실행시 prompt 디버깅
class PromptDebugger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n".join(prompts))
        print("*" * 100)


class RAGChain:
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
            RunnableParallel(
                {
                    "context": retriever | self.format_contexts,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
        )

    # contxt formatt
    def format_contexts(self, contexts):
        return "\n\n".join([d.page_content.strip() for d in reversed(contexts)])
