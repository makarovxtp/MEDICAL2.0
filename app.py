from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

import os
import json

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "medical-chatbot"

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = docsearch.as_retriever(search_kwargs={'k': 2})

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# global state (simple for single-user demo)
chat_history = []
session_stats = {
    "user_messages": 0,
    "risks": {"low": 0, "moderate": 0, "high": 0}
}


# ---------- HELPERS ----------

def format_chat_history_for_prompt(history):
    """Convert LangChain chat_history into a plain-text conversation log."""
    lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def build_sources_html(context_docs):
    """Build HTML list of sources from retrieved documents."""
    if not context_docs or not isinstance(context_docs, list):
        return ""

    sources = []
    for doc in context_docs:
        meta = getattr(doc, "metadata", {}) or {}
        src = (
            meta.get("source")
            or meta.get("file_name")
            or meta.get("id")
            or "Document"
        )
        if src not in sources:
            sources.append(src)

    if not sources:
        return ""

    li_html = "".join(f"<li>{s}</li>" for s in sources)
    html = f"""
<div class="sources-block">
  <div class="sources-title">Sources</div>
  <ul class="sources-list">
    {li_html}
  </ul>
</div>
""".strip()
    return html


# ---------- TRIAGE HELPER ----------

def get_risk_assessment(user_input: str, answer: str) -> dict:
    """
    Ask the LLM to classify the risk of the described symptoms.
    Returns dict: {risk: low/moderate/high, title, advice}
    """
    triage_prompt = f"""
You are a cautious medical triage assistant. You will NOT diagnose diseases.
Your job is ONLY to estimate a basic risk level and give simple guidance.

Classify the user's situation into one of these exact risk levels:
- "low"
- "moderate"
- "high"

Rules:
- Use "high" only for potentially emergency symptoms (severe chest pain, difficulty breathing,
  loss of consciousness, sudden severe headache, signs of stroke, heavy bleeding, etc.).
- Use "moderate" for worrying but not obviously life-threatening symptoms that should be
  checked by a doctor soon.
- Use "low" for mild, common issues that can usually be managed at home.

Return a short JSON object with EXACTLY these keys:
- "risk": "low" | "moderate" | "high"
- "title": a short friendly title in 3–7 words
- "advice": one or two short sentences of guidance

Do not add any extra text before or after the JSON.
Do not use markdown.

User message:
{user_input}

Assistant answer (for reference, do NOT repeat it):
{answer}
    """.strip()

    triage_response = llm.invoke(triage_prompt)
    content = getattr(triage_response, "content", str(triage_response))

    try:
        data = json.loads(content)
    except Exception:
        data = {
            "risk": "low",
            "title": "General information",
            "advice": "This appears to be low-risk, but please consult a doctor if symptoms are severe, persistent, or worrying."
        }

    risk = str(data.get("risk", "low")).lower()
    if risk not in ["low", "moderate", "high"]:
        risk = "low"
    data["risk"] = risk
    return data


# ---------- EMOTION / EMPATHY HELPER ----------

def get_empathy_prefix(user_input: str) -> str:
    """
    Generate a short empathetic preface (1–2 sentences) based on user's emotional tone.
    If the user sounds neutral, return an empty string.
    """
    empathy_prompt = f"""
You are a kind, calm medical assistant.
Read the user's message and decide if they sound anxious, scared, frustrated, or sad.

User message:
{user_input}

If they sound emotionally distressed, respond with 1–2 short, empathetic sentences
in simple language, acknowledging their feelings and reassuring them.

If they sound neutral (just asking information), respond with EXACTLY the word "NONE".

Do NOT give medical advice here. Only show empathy if needed.
    """.strip()

    resp = llm.invoke(empathy_prompt)
    content = getattr(resp, "content", str(resp)).strip()
    if content.upper() == "NONE":
        return ""
    return content


# ---------- SYMPTOM FLOWCHART MODE (GUIDED CHECK) ----------

def get_flow_response(user_input: str, history) -> str:
    """
    Guided symptom check mode: ask focused follow-up questions instead of long answers.
    """
    history_text = format_chat_history_for_prompt(history)
    prompt = f"""
You are running a guided symptom check with the user.

Conversation so far:
{history_text}

New user message:
{user_input}

Your job:
- Ask short, focused follow-up questions one at a time
  to better understand the symptoms (location, duration, severity, triggers, associated symptoms, etc.).
- Keep each message brief: either 1–2 questions at once at most.
- If you already have enough information for a basic suggestion, you may give a short summary and
  simple guidance, then say clearly that they should talk to a real doctor for diagnosis.

Important:
- Do NOT give long explanations in this mode.
- Do NOT mention that you are an AI or language model.
- Keep it conversational and easy to understand.
    """.strip()

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# ---------- MEDICATION SAFETY CHECKER ----------

def get_med_safety_response(user_input: str) -> str:
    """
    Medication safety mode: focus on safe use, interactions, red flags.
    """
    prompt = f"""
You are a medication safety assistant.
User is asking about medicines, doses, or combinations.

User question:
{user_input}

Your behaviour:
- Focus on safety: interactions, overdose risk, age/pregnancy precautions, chronic diseases (like diabetes, kidney disease, liver disease).
- Use clear bullet points where helpful.
- Use simple language.
- Frequently remind: "This is not a prescription. Please consult a doctor or pharmacist before starting or changing medication."
- If something sounds dangerous or unclear, be very cautious and recommend medical consultation.

Give a concise but informative answer.
    """.strip()

    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))


# ---------- DOCTOR SUMMARY GENERATOR ----------

def generate_doctor_summary(history) -> str:
    """
    Use the full chat history to create a structured summary for a doctor.
    """
    if not history:
        return "<div class='summary-card'><p>Not enough information yet to generate a meaningful summary.</p></div>"

    conv_text = format_chat_history_for_prompt(history)
    prompt = f"""
You are a medical assistant helping a doctor.
You will receive the full conversation between a patient and a chatbot.

Conversation:
{conv_text}

Create a concise summary suitable for a doctor before consultation.
Structure the summary with these sections:

1. Main complaints (bullet list)
2. Symptom details (onset, duration, severity, triggers, relieving/aggravating factors if available)
3. Relevant history / risk factors mentioned (e.g., diabetes, hypertension, medications, allergies)
4. Any red-flag symptoms mentioned (if any)
5. Provisional directions for further evaluation (which specialties or tests to consider)
6. Advice already given in the chat (short recap)

Return valid HTML using headings (<h4>) and <ul>/<li> where appropriate.
Do not use markdown.
Do not include any disclaimers at the end.
    """.strip()

    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp))
    return f"<div class='summary-card'>{content}</div>"


# ---------- HEALTH DASHBOARD (SESSION STATS) ----------

def generate_dashboard_html() -> str:
    total = session_stats["user_messages"]
    low = session_stats["risks"]["low"]
    moderate = session_stats["risks"]["moderate"]
    high = session_stats["risks"]["high"]

    if total == 0:
        return """
<div class="health-dashboard-card">
  <h4>Session health overview</h4>
  <p class="health-note">Start chatting to see your session health summary here.</p>
</div>
""".strip()

    # NEW SCORING:
    # weights: low = 1.0, moderate = 0.5, high = 0.1
    weighted_sum = (low * 1.0) + (moderate * 0.5) + (high * 0.1)
    score_raw = weighted_sum / float(total)
    score = int(round(score_raw * 100))

    if score < 0:
        score = 0
    if score > 100:
        score = 100

    html = f"""
<div class="health-dashboard-card">
  <h4>Session health overview</h4>

  <div class="health-metrics-row">
    <div class="metric-block">
      <div class="metric-label">User messages</div>
      <div class="metric-value">{total}</div>
    </div>
    <div class="metric-block">
      <div class="metric-label">Health score</div>
      <div class="metric-value">{score}/100</div>
    </div>
  </div>

  <div class="risk-bars">
    <div class="risk-row">
      <span class="risk-name low">Low risk</span>
      <span class="risk-count">{low}</span>
    </div>
    <div class="risk-row">
      <span class="risk-name moderate">Moderate</span>
      <span class="risk-count">{moderate}</span>
    </div>
    <div class="risk-row">
      <span class="risk-name high">High</span>
      <span class="risk-count">{high}</span>
    </div>
  </div>

  <p class="health-note">
    This is a rough, AI-generated overview and not a clinical tool.
  </p>
</div>
""".strip()
    return html


# ---------- SIMPLIFY (EXPLAIN SIMPLY) ----------

def simplify_text(text: str) -> str:
    prompt = f"""
You are a friendly explainer.
Simplify the following medical answer so that a 12-year-old can understand it,
using very simple words and short sentences. Do not add new medical facts.

Text:
{text}
    """.strip()

    resp = llm.invoke(prompt)
    content = getattr(resp, "content", str(resp))
    html = f"""
<div class="simplified-card">
  <div class="simplified-title">Simplified explanation</div>
  <div class="simplified-text">{content}</div>
</div>
""".strip()
    return html


# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    global chat_history, session_stats

    if request.method != "POST":
        return jsonify({"error": "Invalid request method"})

    msg = request.form.get("msg", "")
    mode = request.form.get("mode", "chat")  # "chat" | "flow" | "med"
    user_input = msg.strip()
    if not user_input:
        return ""

    print("MODE:", mode, "| USER:", user_input)

    context_docs = None

    # choose mode
    if mode == "flow":
        answer = get_flow_response(user_input, chat_history)
    elif mode == "med":
        answer = get_med_safety_response(user_input)
    else:
        # default: RAG chat
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        answer = result["answer"]
        context_docs = result.get("context", None)

    # empathy prefix
    empathy_prefix = get_empathy_prefix(user_input)
    if empathy_prefix:
        full_answer_text = empathy_prefix + "\n\n" + answer
    else:
        full_answer_text = answer

    # triage
    triage = get_risk_assessment(user_input, full_answer_text)
    risk = triage.get("risk", "low")
    title = triage.get("title", "")
    advice = triage.get("advice", "")

    # update stats
    session_stats["user_messages"] += 1
    if risk in session_stats["risks"]:
        session_stats["risks"][risk] += 1

    triage_html = f"""
<div class="triage-card triage-{risk}">
  <div class="triage-header-row">
    <span class="triage-dot triage-dot-{risk}"></span>
    <span class="triage-label">{risk.capitalize()} risk</span>
  </div>
  <div class="triage-title">{title}</div>
  <div class="triage-advice">{advice}</div>
</div>
""".strip()

    sources_html = ""
    if context_docs:
        sources_html = build_sources_html(context_docs)

    final_html = (
        triage_html
        + f"<div class='bot-answer-text'>{full_answer_text}</div>"
        + sources_html
    )

    # update chat history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=full_answer_text))

    print("Response:", full_answer_text[:160], "...")
    return final_html


@app.route("/summary", methods=["POST"])
def summary():
    global chat_history
    html = generate_doctor_summary(chat_history)
    return html


@app.route("/dashboard", methods=["POST"])
def dashboard():
    html = generate_dashboard_html()
    return html


@app.route("/simplify", methods=["POST"])
def simplify_route():
    text = request.form.get("text", "").strip()
    if not text:
        return ""
    html = simplify_text(text)
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
