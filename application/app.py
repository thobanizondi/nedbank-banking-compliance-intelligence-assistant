import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AzureSearch
from groq import Groq

# Load .env from the project root, regardless of where this script is run from
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vector_store = AzureSearch(
    azure_search_endpoint=search_endpoint,
    azure_search_key=search_key,
    index_name=search_index_name,
    embedding_function=embeddings.embed_query,
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_question(question: str, k: int = 8):
    results = vector_store.similarity_search(query=question, k=k)

    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a banking compliance assistant. Answer the question using the context below.
The context may describe a topic without naming it explicitly — use your judgement to connect the content to the question.
Only say you don't have enough information if the context is genuinely unrelated to the question.

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=1024,
        reasoning_effort="low",
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    question = "What is BCBS 239 about?"
    print(f"\nQuestion: {question}")
    answer = ask_question(question)
    print(f"\nAnswer: {answer}")