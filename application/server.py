import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AzureSearch
from groq import Groq

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

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


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


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Nedbank Compliance Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        header { background: #00543C; color: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; }
        header h1 { font-size: 1.8em; margin-bottom: 5px; }
        header p { font-size: 0.9em; opacity: 0.9; }
        .input-group { display: flex; gap: 10px; margin-bottom: 20px; }
        input { flex: 1; padding: 12px; border: 1.5px solid #ddd; border-radius: 6px; font-size: 1em; }
        input:focus { outline: none; border-color: #00543C; }
        button { background: #00543C; color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-size: 1em; font-weight: 600; }
        button:hover { background: #003D2B; }
        button:disabled { background: #999; cursor: not-allowed; }
        .content { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .notice { padding: 12px 15px; margin-bottom: 15px; border-radius: 6px; font-size: 0.9em; }
        .info { background: #E8F0EC; color: #00543C; border-left: 4px solid #00543C; }
        .warn { background: #FBF6E9; color: #6b5a20; border-left: 4px solid #C4A35A; }
        .questions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
        .qbtn { background: white; border: 1.5px solid #E8F0EC; padding: 12px 15px; border-radius: 6px; cursor: pointer; text-align: left; font-size: 0.9em; color: #00543C; transition: all 0.2s; }
        .qbtn:hover { border-color: #00543C; background: #E8F0EC; }
        #answer-wrap { display: none; background: white; padding: 20px; border-radius: 8px; border-top: 4px solid #C4A35A; margin-top: 20px; }
        #answer-wrap.show { display: block; }
        #answer-label { font-size: 0.8em; text-transform: uppercase; color: #00543C; font-weight: 700; margin-bottom: 10px; letter-spacing: 1px; }
        #answer { line-height: 1.6; font-size: 0.95em; }
        #answer p { margin-bottom: 10px; }
        #answer h2, #answer h3 { color: #00543C; margin: 15px 0 8px; }
        #answer code { background: #E8F0EC; padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }
        .footer { text-align: center; font-size: 0.85em; color: #666; margin-top: 30px; }
        .footer a { color: #00543C; text-decoration: none; font-weight: 600; }
        @media (max-width: 700px) {
            .questions { grid-template-columns: 1fr; }
            header h1 { font-size: 1.4em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏦 Nedbank Compliance Assistant</h1>
            <p>Powered by Azure AI Search, Azure Blob Storage & Groq GPT-OSS</p>
        </header>

        <div class="input-group">
            <input type="text" id="q" placeholder="Ask about BCBS 239, POPIA, Nedbank reports..." onkeydown="if(event.key==='Enter') ask()">
            <button onclick="ask()" id="askBtn">Ask</button>
        </div>

        <div class="content">
            <div class="notice info">✅ 7 banking documents indexed — answers grounded in verified sources only.</div>
            <div class="notice warn">⚠️ Responses do not constitute financial, legal, or regulatory advice.</div>

            <h3 style="margin-bottom: 12px;">Suggested Questions</h3>
            <div class="questions">
                <div class="qbtn" onclick="ask_q('What are the BCBS 239 data governance principles?')">BCBS 239 principles</div>
                <div class="qbtn" onclick="ask_q('What does POPIA say about personal data processing?')">POPIA data processing</div>
                <div class="qbtn" onclick="ask_q('What was Nedbank\\'s headline earnings in 2024?')">2024 Earnings</div>
                <div class="qbtn" onclick="ask_q('What are the key risks Nedbank faced in 2024?')">Key risks 2024</div>
                <div class="qbtn" onclick="ask_q('What does BCBS 239 say about data lineage?')">BCBS 239 lineage</div>
                <div class="qbtn" onclick="ask_q('When was Nedbank founded?')">Nedbank history</div>
            </div>
        </div>

        <div id="answer-wrap">
            <div id="answer-label">Answer</div>
            <div id="answer"></div>
        </div>

        <div class="footer">
            <p>Built with FastAPI · Azure AI Search · Groq GPT-OSS</p>
            <p><a href="https://github.com/thobanizondi" target="_blank">GitHub</a> · <a href="https://datascienceportfol.io/thobanizondi" target="_blank">Portfolio</a></p>
        </div>
    </div>

    <script>
        async function ask() {
            const q = document.getElementById('q').value.trim();
            if (!q) return;
            await fetch_answer(q);
        }

        function ask_q(q) {
            document.getElementById('q').value = q;
            fetch_answer(q);
        }

        async function fetch_answer(q) {
            const btn = document.getElementById('askBtn');
            const wrap = document.getElementById('answer-wrap');
            const answerEl = document.getElementById('answer');

            btn.disabled = true;
            btn.innerText = "Thinking...";
            wrap.classList.add('show');
            answerEl.innerText = "Searching documents...";
            wrap.scrollIntoView({ behavior: 'smooth', block: 'center' });

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q })
                });
                const data = await res.json();
                answerEl.innerHTML = marked.parse(data.answer);
            } catch (err) {
                answerEl.innerText = "Error. Please try again.";
            } finally {
                btn.disabled = false;
                btn.innerText = "Ask";
            }
        }
    </script>
</body>
</html>"""


@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_question(req.question)
    return {"answer": answer}