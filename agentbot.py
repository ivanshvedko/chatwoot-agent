#!/usr/bin/env python3
"""
Chatwoot AgentBot — AI-powered customer support agent.
Replaces Captain AI with self-hosted LLM + case memory + wiki.
Handles 200+ conversations/day with SQLite, TF-IDF similarity, and LRU caching.

Architecture:
  Chatwoot → webhook → AgentBot → [Wiki search + Case search] → LLM → Chatwoot API reply
  Resolved conversations → auto-saved as new cases → future queries find them

LLM: Ollama Cloud (deepseek-v4-pro, primary) → BlockRun (deepseek-v4-flash, fallback)
"""

import os
import sys
import json
import sqlite3
import hashlib
import hmac
import time
import re
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional

import requests
import numpy as np
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("agentbot")

# ── Configuration (env vars with defaults) ────────────────────────────────────
CHATWOOT_URL = os.getenv("CHATWOOT_URL", "http://host.docker.internal:3000")
CHATWOOT_ACCESS_TOKEN = os.getenv("CHATWOOT_ACCESS_TOKEN", "")
CHATWOOT_ACCOUNT_ID = int(os.getenv("CHATWOOT_ACCOUNT_ID", "1"))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# Ollama Cloud — primary LLM
OLLAMA_CLOUD_KEY = os.getenv("OLLAMA_CLOUD_KEY", "")
OLLAMA_CLOUD_URL = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com/v1/chat/completions")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "deepseek-v4-pro")

# BlockRun — fallback LLM (free, no key needed)
BLOCKRUN_URL = os.getenv("BLOCKRUN_URL", "https://blockrun.ai/api/v1/chat/completions")
BLOCKRUN_MODEL = os.getenv("BLOCKRUN_MODEL", "nvidia/deepseek-v4-flash")

# Storage
DB_PATH = os.getenv("DB_PATH", "/data/cases.db")
WIKI_PATH = os.getenv("WIKI_PATH", "/data/wiki")

# Performance
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "500"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "15"))
MAX_WIKI_ARTICLES = int(os.getenv("MAX_WIKI_ARTICLES", "2"))
MAX_SIMILAR_CASES = int(os.getenv("MAX_SIMILAR_CASES", "3"))

# ── Initialization ───────────────────────────────────────────────────────────
DetectorFactory.seed = 0
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(WIKI_PATH, exist_ok=True)

app = FastAPI(title="Chatwoot AgentBot", version="1.0.0")

# ── Database ─────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_text TEXT NOT NULL,
                solution_text TEXT NOT NULL,
                problem_hash TEXT UNIQUE,
                language TEXT,
                tags TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_hash ON cases(problem_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_lang ON cases(language)")
        conn.commit()
    logger.info(f"Database initialized: {DB_PATH}")

init_db()

# ── LRU Cache ────────────────────────────────────────────────────────────────
class LRUCache:
    """Thread-safe-ish LRU cache with TTL for LLM responses."""
    def __init__(self, max_size: int = 500, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return value

    def set(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, time.time())
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.cache)

llm_cache = LRUCache(max_size=CACHE_SIZE, ttl=CACHE_TTL)

# ── Rate Limiter ─────────────────────────────────────────────────────────────
class RateLimiter:
    """Simple in-memory rate limiter per conversation."""
    def __init__(self, max_per_minute: int = 5):
        self.windows: dict[int, list[float]] = {}
        self.max_per_minute = max_per_minute

    def allow(self, conversation_id: int) -> bool:
        now = time.time()
        window = self.windows.get(conversation_id, [])
        # Remove entries older than 60s
        window = [t for t in window if now - t < 60]
        if len(window) >= self.max_per_minute:
            self.windows[conversation_id] = window
            return False
        window.append(now)
        self.windows[conversation_id] = window
        return True

rate_limiter = RateLimiter(max_per_minute=5)

# ── Language Detection ───────────────────────────────────────────────────────
LANG_NAMES = {
    "ru": "Russian", "es": "Spanish", "en": "English",
    "pt": "Portuguese", "fr": "French", "de": "German", "it": "Italian"
}

def detect_language(text: str) -> str:
    """Detect language; returns ISO code, defaults to 'en'."""
    try:
        lang = detect(text)
        return lang if lang in LANG_NAMES else "en"
    except Exception:
        return "en"

# ── Greeting / Short Message Detection ──────────────────────────────────────
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hola|привет|здравствуй|здравствуйте|buenos días|buenas tardes"
    r"|bonjour|salut|ciao|hallo|oi|olá)[\s!.,]*$",
    re.IGNORECASE
)
THANKS_PATTERNS = re.compile(
    r"^(thanks|thank you|спасибо|gracias|merci|danke|obrigado|grazie)[\s!.,]*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_PATTERNS.match(text.strip()))

def is_thanks(text: str) -> bool:
    return bool(THANKS_PATTERNS.match(text.strip()))

# ── Wiki Search ──────────────────────────────────────────────────────────────
def search_wiki(query: str, top_k: int = None) -> list[dict]:
    """Search wiki .md files by keyword relevance."""
    if top_k is None:
        top_k = MAX_WIKI_ARTICLES

    wiki_dir = Path(WIKI_PATH)
    if not wiki_dir.exists() or not any(wiki_dir.iterdir()):
        return []

    articles = []
    for md_file in wiki_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.stem
            articles.append({"title": title, "content": content, "file": str(md_file)})
        except Exception:
            continue

    if not articles:
        return []

    query_lower = query.lower()
    scored = []
    for art in articles:
        content_lower = art["content"].lower()
        score = 0
        for word in query_lower.split():
            if len(word) > 2:
                score += content_lower.count(word)
        # Bonus for title match
        if any(w in art["title"].lower() for w in query_lower.split() if len(w) > 2):
            score += 5
        if score > 0:
            scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [art for _, art in scored[:top_k]]

# ── Case Similarity Search ───────────────────────────────────────────────────
# Global TF-IDF state — rebuilt periodically
_tfidf_state: dict = {"vectorizer": None, "matrix": None, "cases": [], "built_at": 0}

def _rebuild_tfidf(language: str):
    """Rebuild TF-IDF index for a given language."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, problem_text, solution_text, tags, usage_count
               FROM cases
               WHERE language = ? OR language = 'unknown'
               ORDER BY usage_count DESC
               LIMIT 500""",
            (language,)
        ).fetchall()

    if len(rows) < 2:
        _tfidf_state["vectorizer"] = None
        _tfidf_state["matrix"] = None
        _tfidf_state["cases"] = []
        _tfidf_state["built_at"] = time.time()
        return

    problems = [row["problem_text"] for row in rows]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
    matrix = vectorizer.fit_transform(problems)

    _tfidf_state["vectorizer"] = vectorizer
    _tfidf_state["matrix"] = matrix
    _tfidf_state["cases"] = rows
    _tfidf_state["built_at"] = time.time()

def search_similar_cases(query: str, language: str, top_k: int = None) -> list[dict]:
    """Find similar past cases using TF-IDF cosine similarity."""
    if top_k is None:
        top_k = MAX_SIMILAR_CASES

    # Rebuild TF-IDF every 5 minutes
    if time.time() - _tfidf_state["built_at"] > 300:
        _rebuild_tfidf(language)

    vectorizer = _tfidf_state["vectorizer"]
    matrix = _tfidf_state["matrix"]
    cases = _tfidf_state["cases"]

    if vectorizer is None or matrix is None or not cases:
        return []

    try:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, matrix).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim > 0.15:
                row = cases[idx]
                results.append({
                    "id": row["id"],
                    "problem": row["problem_text"],
                    "solution": row["solution_text"],
                    "similarity": sim,
                    "tags": json.loads(row["tags"]),
                    "usage_count": row["usage_count"]
                })
        return results
    except Exception as e:
        logger.warning(f"TF-IDF search error: {e}")
        return []

# ── LLM Call with Fallback ───────────────────────────────────────────────────
def call_llm(system_prompt: str, user_message: str) -> Optional[str]:
    """Call LLM: Ollama Cloud → BlockRun fallback."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # ── Primary: Ollama Cloud ──
    if OLLAMA_CLOUD_KEY:
        try:
            resp = requests.post(
                OLLAMA_CLOUD_URL,
                headers={
                    "Authorization": f"Bearer {OLLAMA_CLOUD_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OLLAMA_CLOUD_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 600
                },
                timeout=LLM_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"Ollama Cloud {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Ollama Cloud exception: {e}")

    # ── Fallback: BlockRun (free) ──
    try:
        resp = requests.post(
            BLOCKRUN_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": BLOCKRUN_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 600
            },
            timeout=LLM_TIMEOUT
        )
        if resp.status_code == 200:
            data = resp.json()
            logger.info("BlockRun fallback used successfully")
            return data["choices"][0]["message"]["content"]
        else:
            logger.error(f"BlockRun {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.error(f"BlockRun exception: {e}")

    return None

# ── System Prompt Builder ────────────────────────────────────────────────────
def build_system_prompt(language: str, wiki_articles: list[dict], similar_cases: list[dict]) -> str:
    """Build context-rich system prompt for the LLM."""
    lang_name = LANG_NAMES.get(language, "English")

    prompt = f"""You are a professional customer support agent serving multiple clients across different websites and projects. You handle technical issues, questions, complaints, and feature requests.

CRITICAL RULES:
• ALWAYS respond in {lang_name} (the user's language)
• Be concise, helpful, and polite — 2-4 sentences is usually enough
• If you don't know the answer, say so and offer to escalate to a human agent
• Use the provided context (knowledge base + past cases) to inform your response
• NEVER make up information you're not confident about
• For technical issues, ask 1-2 clarifying questions if the problem is vague
• For complaints, acknowledge the frustration first, then offer solutions
• For feature requests, thank the user and note that it will be forwarded to the team
• Do NOT mention that you are an AI unless directly asked
"""

    if wiki_articles:
        prompt += "\n─── KNOWLEDGE BASE ───\n"
        for art in wiki_articles:
            prompt += f"\n## {art['title']}\n{art['content'][:800]}\n"

    if similar_cases:
        prompt += "\n─── SIMILAR PAST CASES (use these as reference) ───\n"
        for i, case in enumerate(similar_cases, 1):
            prompt += f"\nCase {i}:\nProblem: {case['problem'][:300]}\nSolution: {case['solution'][:300]}\n"

    return prompt

# ── Chatwoot API Helpers ─────────────────────────────────────────────────────
def _api_headers() -> dict:
    return {"api_access_token": CHATWOOT_ACCESS_TOKEN, "Content-Type": "application/json"}

def send_reply(conversation_id: int, text: str, private: bool = False) -> bool:
    """Post a message to a Chatwoot conversation."""
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    try:
        resp = requests.post(url, headers=_api_headers(), json={
            "content": text, "message_type": "outgoing", "private": private
        }, timeout=10)
        if resp.status_code != 200:
            logger.error(f"Reply failed {resp.status_code}: {resp.text[:200]}")
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Reply exception: {e}")
        return False

def update_conversation_status(conversation_id: int, status: str) -> bool:
    """Change conversation status: open / pending / resolved."""
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}"
    try:
        resp = requests.patch(url, headers=_api_headers(), json={"status": status}, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Status update exception: {e}")
        return False

def get_conversation_messages(conversation_id: int) -> list[dict]:
    """Fetch all messages from a conversation."""
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    try:
        resp = requests.get(url, headers=_api_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json().get("payload", [])
    except Exception as e:
        logger.error(f"Fetch messages exception: {e}")
    return []

# ── Case Persistence ─────────────────────────────────────────────────────────
def save_case(problem: str, solution: str, language: str):
    """Save or update a resolved case in the database."""
    problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
    with sqlite3.connect(DB_PATH) as conn:
        existing = conn.execute("SELECT id FROM cases WHERE problem_hash = ?", (problem_hash,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE cases SET usage_count = usage_count + 1, solution_text = ? WHERE id = ?",
                (solution, existing[0])
            )
        else:
            conn.execute(
                "INSERT INTO cases (problem_text, solution_text, problem_hash, language) VALUES (?, ?, ?, ?)",
                (problem, solution, problem_hash, language)
            )
        conn.commit()

# ── Webhook Verification ─────────────────────────────────────────────────────
def verify_webhook(payload: bytes, signature: str) -> bool:
    """HMAC-SHA256 verification of Chatwoot webhook."""
    if not WEBHOOK_SECRET:
        return True
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

# ── Core: Message Processing ─────────────────────────────────────────────────
def process_message(conversation_id: int, content: str):
    """Full pipeline: detect → search → LLM → reply."""
    logger.info(f"Processing conv={conversation_id}: «{content[:80]}...»")

    # ── Rate limit ──
    if not rate_limiter.allow(conversation_id):
        logger.warning(f"Rate limited conv={conversation_id}")
        return

    # ── Language ──
    language = detect_language(content)

    # ── Greeting / Thanks → fast path (no search, no LLM) ──
    if is_greeting(content):
        replies = {
            "ru": "Здравствуйте! Я виртуальный ассистент поддержки. Опишите ваш вопрос или проблему, и я постараюсь помочь.",
            "es": "¡Hola! Soy el asistente virtual de soporte. Describa su consulta o problema y haré todo lo posible por ayudarle.",
            "en": "Hello! I'm the virtual support assistant. Please describe your question or issue, and I'll do my best to help."
        }
        send_reply(conversation_id, replies.get(language, replies["en"]))
        return

    if is_thanks(content):
        replies = {
            "ru": "Пожалуйста! Если понадобится помощь — обращайтесь.",
            "es": "¡De nada! Si necesita más ayuda, no dude en preguntar.",
            "en": "You're welcome! Feel free to reach out if you need anything else."
        }
        send_reply(conversation_id, replies.get(language, replies["en"]))
        return

    # ── Cache check ──
    cache_key = hashlib.sha256(content.lower().strip().encode()).hexdigest()
    cached = llm_cache.get(cache_key)
    if cached:
        logger.info(f"Cache HIT: {content[:50]}...")
        send_reply(conversation_id, cached)
        return

    # ── Search ──
    wiki_articles = search_wiki(content)
    similar_cases = search_similar_cases(content, language)

    # ── Build prompt & call LLM ──
    system_prompt = build_system_prompt(language, wiki_articles, similar_cases)
    reply = call_llm(system_prompt, content)

    if reply:
        llm_cache.set(cache_key, reply)
        send_reply(conversation_id, reply)
        logger.info(f"Reply sent conv={conversation_id}, lang={language}")
    else:
        # Both LLMs failed — escalate to human
        fallback = {
            "ru": "Извините, возникли технические трудности с обработкой запроса. Человек-оператор подключится в ближайшее время.",
            "es": "Disculpe, tenemos dificultades técnicas. Un agente humano se conectará en breve.",
            "en": "Sorry, I'm experiencing technical difficulties. A human agent will assist you shortly."
        }
        send_reply(conversation_id, fallback.get(language, fallback["en"]))
        update_conversation_status(conversation_id, "open")
        logger.error(f"LLM failed conv={conversation_id}, escalated")

# ── Learning from Resolved Conversations ─────────────────────────────────────
def learn_from_conversation(conversation_id: int):
    """Extract problem→solution pair from a resolved conversation and save it."""
    messages = get_conversation_messages(conversation_id)
    if len(messages) < 2:
        return

    problem = None
    for msg in messages:
        if msg.get("message_type") == "incoming" and msg.get("content"):
            problem = msg["content"].strip()
            break

    solution = None
    for msg in reversed(messages):
        if msg.get("message_type") == "outgoing" and msg.get("content") and not msg.get("private"):
            solution = msg["content"].strip()
            break

    if problem and solution and len(problem) > 10 and len(solution) > 10:
        language = detect_language(problem)
        save_case(problem, solution, language)
        logger.info(f"Learned conv={conversation_id}: {problem[:50]}... → {solution[:50]}...")

# ── FastAPI Endpoints ────────────────────────────────────────────────────────
@app.post("/chatwoot-webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Main webhook endpoint for Chatwoot AgentBot integration."""
    body = await request.body()
    signature = request.headers.get("X-Chatwoot-Signature", "")

    if not verify_webhook(body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    data = json.loads(body)
    event = data.get("event", "")
    logger.info(f"Webhook received: event={event}, keys={list(data.keys())}")
    # Dump full payload for debugging
    logger.info(f"Webhook payload: {json.dumps(data, default=str)[:500]}")

    # ── New message from customer ──
    if event == "message_created":
        # Chatwoot sends message_type and sender at top level, not nested
        message_type = data.get("message_type", "")
        sender = data.get("sender", {})
        conversation = data.get("conversation", {})
        logger.info(f"message_created: msg_type={message_type}, sender_type={sender.get('type')}")

        if message_type != "incoming":
            logger.info(f"Ignored: message_type={message_type} (not incoming)")
            return {"status": "ignored", "reason": "not incoming"}
        # Skip messages from the bot itself (prevent loops)
        if sender.get("name") == "AI Assistant" or data.get("private"):
            logger.info(f"Ignored: from bot or private")
            return {"status": "ignored", "reason": "from bot"}

        conversation_id = conversation.get("id")
        content = (data.get("content") or "").strip()

        if not content or not conversation_id:
            return {"status": "ignored", "reason": "empty"}

        background_tasks.add_task(process_message, conversation_id, content)
        return {"status": "processing"}

    # ── Conversation resolved → learn ──
    elif event == "conversation_status_changed":
        new_status = data.get("status", "")
        if new_status == "resolved":
            conversation_id = data.get("conversation", {}).get("id")
            if conversation_id:
                background_tasks.add_task(learn_from_conversation, conversation_id)
        return {"status": "ok"}

    return {"status": "ignored", "reason": f"unhandled event: {event}"}

@app.get("/health")
def health():
    """Health check endpoint."""
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    return {
        "status": "ok",
        "cases_count": total,
        "cache_entries": len(llm_cache),
        "tfidf_built_at": _tfidf_state["built_at"]
    }

@app.get("/stats")
def stats():
    """Detailed statistics."""
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        by_lang = conn.execute("SELECT language, COUNT(*) FROM cases GROUP BY language").fetchall()
        top = conn.execute(
            "SELECT problem_text, usage_count FROM cases ORDER BY usage_count DESC LIMIT 10"
        ).fetchall()
    return {
        "total_cases": total,
        "by_language": {lang: cnt for lang, cnt in by_lang},
        "top_queries": [{"problem": p[:80], "count": c} for p, c in top],
        "cache_size": len(llm_cache)
    }

# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Chatwoot AgentBot on port 5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
