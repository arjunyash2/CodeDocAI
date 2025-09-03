from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.retriever import get_retriever
from app.llm_wrapper import get_llm

def create_agent():
    """
    Creates and configures a RetrievalQA agent to answer questions
    based on the documents in the vector store.
    """
    retriever = get_retriever()
    llm = get_llm()

    # --- KEY CHANGE ---
    # Added a clear instruction for the model to follow if the answer is not in the context.
    # This helps prevent the model from "hallucinating" or making up answers.
    template = """
Based on this context:
{context}

Answer this question: {question}

If the answer is not found in the context, say "I could not find the answer in the provided documents."
"""

    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa

