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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nedbank Compliance Assistant | Azure RAG</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.0/marked.min.js"></script>
        <style>
            :root {
                --green: #00543C;
                --green-dark: #003D2B;
                --green-light: #E8F0EC;
                --gold: #C4A35A;
                --bg: #F4F6F5;
                --sidebar-w: 280px;
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background: var(--bg);
                color: #1a1a1a;
                display: flex;
                min-height: 100vh;
            }

            #sidebar {
                width: var(--sidebar-w);
                background: var(--green);
                color: white;
                padding: 28px 20px;
                position: fixed;
                height: 100vh;
                overflow-y: auto;
            }
            #sidebar h2 {
                font-size: 1.15em;
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 4px;
            }
            #sidebar .stack-label {
                font-size: 0.78em;
                color: #cfe3d8;
                margin-bottom: 24px;
            }
            .doc-item {
                background: rgba(255,255,255,0.08);
                border-radius: 8px;
                padding: 10px 12px;
                margin-bottom: 8px;
                font-size: 0.85em;
                display: flex;
                gap: 10px;
                align-items: flex-start;
            }
            .doc-item .icon { font-size: 1.2em; }
            .doc-item .title { font-weight: 600; }
            .doc-item .sub { color: #cfe3d8; font-size: 0.85em; }
            #sidebar h3 {
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin: 22px 0 12px;
                color: var(--gold);
            }

            #main {
                margin-left: var(--sidebar-w);
                flex: 1;
                padding: 0 0 60px;
            }
            header {
                background: linear-gradient(135deg, var(--green), var(--green-dark));
                color: white;
                padding: 36px 40px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 16px;
            }
            header .title-row { display: flex; align-items: center; gap: 14px; }
            header .icon-badge {
                background: white;
                border-radius: 50%;
                width: 52px; height: 52px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.6em;
            }
            header h1 { font-size: 1.5em; font-weight: 700; }
            header p { font-size: 0.88em; color: #d9e8e0; margin-top: 4px; }
            .badge {
                background: rgba(255,255,255,0.15);
                border: 1px solid rgba(255,255,255,0.35);
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.82em;
                white-space: nowrap;
            }

            #input-bar {
                background: white;
                padding: 20px 40px;
                display: flex;
                gap: 10px;
                border-bottom: 1px solid #e0e4e2;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            #q {
                flex: 1;
                padding: 13px 16px;
                border: 1.5px solid #d0d7d4;
                border-radius: 8px;
                font-size: 0.95em;
                outline: none;
            }
            #q:focus { border-color: var(--green); }
            #askBtn {
                background: var(--green);
                color: white;
                border: none;
                padding: 0 28px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                font-weight: 600;
                transition: background 0.2s;
                white-space: nowrap;
            }
            #askBtn:hover { background: var(--green-dark); }
            #askBtn:disabled { background: #a3b8ae; cursor: not-allowed; }

            .content { padding: 28px 40px; max-width: 1100px; }

            .notice {
                border-radius: 8px;
                padding: 14px 18px;
                font-size: 0.88em;
                margin-bottom: 14px;
            }
            .notice.info {
                background: var(--green-light);
                color: var(--green-dark);
                border-left: 4px solid var(--green);
            }
            .notice.warn {
                background: #FBF6E9;
                color: #6b5a20;
                border-left: 4px solid var(--gold);
            }

            h2.section {
                font-size: 1.1em;
                margin: 26px 0 14px;
                color: var(--green-dark);
            }

            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }
            .qbtn {
                background: white;
                border: 1.5px solid var(--green-light);
                border-radius: 8px;
                padding: 14px 18px;
                text-align: left;
                cursor: pointer;
                font-size: 0.92em;
                color: var(--green-dark);
                transition: all 0.15s;
            }
            .qbtn:hover {
                border-color: var(--green);
                background: var(--green-light);
                transform: translateY(-1px);
            }

            #answer-wrap {
                margin-top: 22px;
                display: none;
                background: white;
                border-radius: 10px;
                border-top: 4px solid var(--gold);
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
                padding: 22px 26px;
            }
            #answer-label {
                font-size: 0.78em;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--green);
                font-weight: 700;
                margin-bottom: 10px;
            }
            #answer {
                line-height: 1.65;
                font-size: 0.97em;
            }
            #answer table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.92em; }
            #answer th, #answer td { border: 1px solid #dde3e0; padding: 8px 10px; text-align: left; vertical-align: top; }
            #answer th { background: var(--green-light); color: var(--green-dark); }
            #answer h1, #answer h2, #answer h3 { color: var(--green-dark); margin: 16px 0 8px; }
            #answer strong { color: var(--green-dark); }
            #answer ul, #answer ol { margin: 8px 0 8px 20px; }
            #answer p { margin-bottom: 10px; }
            #answer code { background: var(--green-light); padding: 2px 5px; border-radius: 4px; font-size: 0.9em; }

            .stack-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 40px;
                padding-top: 18px;
                border-top: 1px solid #e0e4e2;
                font-size: 0.82em;
                color: #667;
            }
            .stack-footer a { color: var(--green); text-decoration: none; font-weight: 600; }

            @media (max-width: 800px) {
                #sidebar { display: none; }
                #main { margin-left: 0; }
                .grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h2>🏦 Nedbank Assistant</h2>
            <div class="stack-label">Azure AI Search &middot; Blob Storage &middot; Groq</div>

            <h3>Loaded Documents</h3>
            <div class="doc-item"><span class="icon">📊</span><div><div class="title">2024 Integrated Report</div><div class="sub">Strategy &amp; Governance</div></div></div>
            <div class="doc-item"><span class="icon">💰</span><div><div class="title">2024 Annual Results</div><div class="sub">Financial Results</div></div></div>
            <div class="doc-item"><span class="icon">📄</span><div><div class="title">2024 Financial Statements</div><div class="sub">Audited Financials</div></div></div>
            <div class="doc-item"><span class="icon">🔒</span><div><div class="title">POPIA</div><div class="sub">Regulatory Compliance</div></div></div>
            <div class="doc-item"><span class="icon">🏛️</span><div><div class="title">BCBS 239</div><div class="sub">Data Governance</div></div></div>
            <div class="doc-item"><span class="icon">📋</span><div><div class="title">Nedbank Factsheet</div><div class="sub">Company Overview</div></div></div>
            <div class="doc-item"><span class="icon">📜</span><div><div class="title">Nedbank Firsts</div><div class="sub">Corporate History</div></div></div>

            <h3>Architecture</h3>
            <div class="doc-item"><span class="icon">☁️</span><div><div class="title">Azure Blob Storage</div><div class="sub">Document source</div></div></div>
            <div class="doc-item"><span class="icon">🔍</span><div><div class="title">Azure AI Search</div><div class="sub">Vector retrieval</div></div></div>
            <div class="doc-item"><span class="icon">⚡</span><div><div class="title">Groq GPT-OSS</div><div class="sub">Answer generation</div></div></div>
        </div>

        <div id="main">
            <header>
                <div class="title-row">
                    <div class="icon-badge">🏦</div>
                    <div>
                        <h1>Nedbank Banking Compliance &amp; Intelligence Assistant</h1>
                        <p>Powered by 7 verified Nedbank documents &middot; POPIA &middot; BCBS 239 &middot; Azure-native RAG</p>
                    </div>
                </div>
                <div class="badge">🔒 Secure &middot; Document-Grounded</div>
            </header>

            <div id="input-bar">
                <input id="q" placeholder="Ask about banking compliance, POPIA, BCBS 239, or Nedbank reports..." onkeydown="if(event.key==='Enter') ask()"/>
                <button id="askBtn" onclick="ask()">Ask</button>
            </div>

            <div class="content">
                <div class="notice info">✅ 7 banking documents indexed via Azure AI Search — answers are grounded in these documents only.</div>
                <div class="notice warn"><strong>Security Notice:</strong> This assistant only answers based on the loaded documents. Responses do not constitute financial, legal, or regulatory advice. Always verify with qualified professionals before making compliance or investment decisions.</div>

                <h2 class="section">Suggested Compliance Questions</h2>
                <div class="grid">
                    <div class="qbtn" onclick="fillAndAsk('What are the BCBS 239 data governance principles?')">What are the BCBS 239 data governance principles?</div>
                    <div class="qbtn" onclick="fillAndAsk('What does POPIA say about personal data processing?')">What does POPIA say about personal data processing?</div>
                    <div class="qbtn" onclick="fillAndAsk('What was Nedbank\\'s headline earnings in 2024?')">What was Nedbank's headline earnings in 2024?</div>
                    <div class="qbtn" onclick="fillAndAsk('What are the key risks Nedbank faced in 2024?')">What are the key risks Nedbank faced in 2024?</div>
                    <div class="qbtn" onclick="fillAndAsk('What does BCBS 239 say about data lineage?')">What does BCBS 239 say about data lineage?</div>
                    <div class="qbtn" onclick="fillAndAsk('When was Nedbank founded?')">When was Nedbank founded?</div>
                </div>

                <div id="answer-wrap">
                    <div id="answer-label">Answer</div>
                    <div id="answer"></div>
                </div>

                <div class="stack-footer">
                    <span>Document-grounded responses only</span>
                    <span>Built with Python &middot; FastAPI &middot; Azure AI Search &middot; Azure Blob Storage &middot; Groq GPT-OSS</span>
                    <span><a href="https://github.com/thobanizondi" target="_blank">GitHub</a> &middot; <a href="https://datascienceportfol.io/thobanizondi" target="_blank">Portfolio</a></span>
                </div>
            </div>
        </div>

        <script>
        async function ask() {
            const q = document.getElementById('q').value.trim();
            if (!q) return;
            const btn = document.getElementById('askBtn');
            const wrap = document.getElementById('answer-wrap');
            const answerEl = document.getElementById('answer');

            btn.disabled = true;
            btn.innerText = "Thinking...";
            wrap.style.display = 'block';
            answerEl.innerText = "Searching documents and generating an answer...";
            wrap.scrollIntoView({behavior: 'smooth', block: 'center'});

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                answerEl.innerHTML = marked.parse(data.answer);
            } catch (err) {
                answerEl.innerText = "Something went wrong. Please try again.";
            } finally {
                btn.disabled = false;
                btn.innerText = "Ask";
            }
        }

        function fillAndAsk(question) {
            document.getElementById('q').value = question;
            ask();
        }
        </script>
    </body>
    </html>
    """


@app.post("/ask")
def ask(req: QuestionRequest):
    answer = ask_question(req.question)
    return {"answer": answer}