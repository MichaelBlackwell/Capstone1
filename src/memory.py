import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from src.retriever import get_retriever
from src.chains import get_llm

load_dotenv()


CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up
question to be a standalone question that includes relevant context from the chat history.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful business data assistant for a sales analytics platform.
Answer the user's question using ONLY the data context provided. Cite specific numbers
when available. If you cannot answer from the context, say so.

Data Context:
{context}

Question: {question}

Answer:"""
)


def create_memory():
    """Create a conversation memory instance."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


def build_conversational_chain(memory=None, model="openai/gpt-oss-120b"):
    """Build a ConversationalRetrievalChain combining RAG + memory."""
    llm = get_llm(model)
    retriever = get_retriever(k=3)

    if memory is None:
        memory = create_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    return chain, memory


class ChatSession:
    """Manages a conversational RAG session with persistent memory."""

    def __init__(self, model="openai/gpt-oss-120b"):
        self.chain, self.memory = build_conversational_chain(model=model)

    def ask(self, question: str) -> str:
        """Send a message and get a response with conversation context."""
        result = self.chain.invoke({"question": question})
        return result["answer"]

    def get_history(self) -> list:
        """Return the conversation history as a list of messages."""
        return self.memory.chat_memory.messages

    def clear(self):
        """Clear conversation history."""
        self.memory.clear()


if __name__ == "__main__":
    session = ChatSession()

    questions = [
        "What product has the highest total sales?",
        "How does it compare to the lowest-selling product?",
        "What about regional performance for that top product?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        answer = session.ask(q)
        print(f"Assistant: {answer}")

    print(f"\n{'='*60}")
    print(f"Conversation history: {len(session.get_history())} messages")
