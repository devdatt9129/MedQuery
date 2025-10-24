
# 🩺 MedQuery Chat – Clinical QA Assistant

**MedQuery Chat** is a clinical QA system that allows healthcare providers to query patient encounter transcripts and SOAP notes using natural language.  
It uses a retrieval-augmented generation (RAG) pipeline to generate grounded, explainable responses with citations.

---


## 📁 Project Structure
* **project/**
    * **backend/**
        * **app/**
            * `__init__.py`
            * `main.py` `# FastAPI entrypoint`
            * `rag.py` `# RAG pipeline logic`
            * `retrieval.py` `# Retrieval-specific logic`
            * `settings.py`
            * **data/**
                * `soap_note.json`
                * `transcript.json`
        * **chroma_db/** `# ChromaDB persistent storage`
        * `Dockerfile` `# Backend Dockerfile`
        * `requirements.txt`
    * **ds-frontend/**
        * **app/**
            * `favicon.ico`
            * `globals.css`
            * `layout.tsx` `# Next.js root layout`
            * `page.tsx` `# Chat UI`
        * **public/** `# Static assets`
        * `.env.local` `# Local environment variables`
        * `.gitignore`
        * `Dockerfile` `# Frontend Dockerfile`
        * `next-env.d.ts`
        * `next.config.ts`
        * `package.json`
        * `postcss.config.mjs`
        * `README.md`
        * `tsconfig.json`
    * `.env.example`
    * `.gitignore`
    * `render.yaml`

## 🚀 Features

### 🧠 Retrieval-Augmented Generation (RAG)
- Embeds transcripts + SOAP notes into Chroma vector store.
- Retrieves relevant context using dense (OpenAI/e5) + BM25 hybrid retrieval.
- Uses LangChain `ConversationalRetrievalChain` for context-aware chat.
- Supports follow-ups through per-session Redis memory.

### 💬 Clinical Chat Interface
- Modern chat UI built with **Next.js + TailwindCSS**.
- Expandable citations under each answer.
- Session persistence with Redis.
- Rotating placeholder tips to guide user queries.

### ⚙️ Infrastructure
- **Backend** → FastAPI on **Render**
- **Frontend** → Next.js on **Vercel**
- **Memory** → Redis (Upstash)
- **Vector DB** → Chroma (Render disk mount)

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | Next.js 16 (App Router), TailwindCSS |
| Backend | FastAPI, LangChain, ChromaDB |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o-mini |
| Memory | Redis (Upstash) |
| Hosting | Vercel (frontend), Render (backend) |

---


## 🧠 Design Choices & Why They’re Good for This Challenge

This section explains **what I used** and **why it’s a good fit** for a clinical QA demo that must:
- answer follow-ups **with context**,
- handle **long transcripts**,
- be **reliable, fast, and cheap** to run,
- and be **easy to extend** later.

---

### 1) Python + FastAPI + LangChain (backend)

**Why**
- **Python** is the de-facto ecosystem for NLP/RAG; the libraries you need (LangChain, tokenizers, vector DB clients) are mature and well-supported.
- **FastAPI** gives a clean, typed, high-performance HTTP API (async-ready, automatic docs).
- **LangChain** gives production-ready building blocks I’d otherwise re-implement:
  - chunking/splitters,
  - retrievers & composable chains,
  - memory adapters (Redis),
  - pluggable LLMs/embeddings.

**What this buys us**
- Quicker iteration; fewer chances to write buggy plumbing.
- Swap models/datastores in **1–2 lines** instead of rewriting pipelines.
- Easier to demonstrate **code quality & modularity** for the challenge rubric.

**Why not “DIY only” or a different web framework?**
- Re-implementing retrieval/memory is error-prone and slow.
- Flask/Django are fine, but FastAPI is faster to type, async-friendly, and great DX for APIs.

---

### 2) Next.js (frontend) + simple chat UI

**Why**
- Built-in routing, SSR/CSR balance, and great DX for quickly shipping a polished chat.
- Deployed on **Vercel** in minutes; environment variables for clean separation from backend.

**Why not a bare HTML app?**
- We’d lose modern DX, environment management, and future extensibility (auth, uploads, etc.).

---

### 3) Retrieval-Augmented Generation (RAG) with **ensemble retrieval**
- **Dense** retrieval via embeddings (semantic match).
- **BM25** (lexical) as a safety net for exact phrases (e.g., drug names, lab acronyms).
- I **merge & dedupe** results → better recall on both “fuzzy” and “literal” queries.

**Why**
- Long transcripts don’t fit into model context.
- Providers ask both fuzzy (“triggers?”) and literal (“what tests?”) questions.
- Ensemble retrieval reduces “misses” and keeps answers grounded.

**Why not only dense OR only BM25?**
- Dense-only can miss exact terms; BM25-only can miss paraphrases/synonyms.
- The combo is small extra cost for noticeably better recall.

---

### 4) **Chroma** as the vector store

**Why**
- **Embeddable & lightweight**, runs locally or on a small server with no external service costs.
- Simple persistence to disk; perfect for a coding challenge or small-scale demo.

**Why not Pinecone / Weaviate / pgvector (cloud)?**
- All are great, but introduce infra & cost. For a **self-contained** demo, Chroma is enough.
- I designed the code so swapping stores is trivial later.

---

### 5) **Redis** for chat history & lightweight memory

**What I use it for**
- **Conversation history**: reliable, shared across processes; survives restarts.
- **Low-latency** lookup for recent turns; ideal for follow-ups like “and what about…?”
- Optional: store **distilled “fact” snippets** (fast ephemeral memory separate from clinical docs).

**Why**
- Fast, simple, battle-tested; one environment variable to point to Upstash/Redis Cloud.
- Avoids memory loss across server restarts and enables **multi-instance** scaling.

**Why not in-process memory only?**
- You’d lose state on each restart/scale event. Redis makes sessions **sticky and durable**.

---

### 6) Embeddings: **OpenAI `text-embedding-3-small`** (default) and optional **E5** (HF)

**Why**
- `text-embedding-3-small` → strong quality/price, multilingual, easy to use.
- **E5** (via sentence-transformers) is offered for **offline/local** or cost-sensitive runs.
- The code keeps embeddings **pluggable** → you can switch with one config flag.

**Why not only one or the other?**
- Flexibility matters: some teams prefer **zero external calls** (E5), others prefer **best QPS/quality** (OpenAI). We support both.

---

### 7) LLM: **GPT-4o-mini** (or similar) with conservative temperature

**Why**
- Great cost/quality/latency trade-off for **short, factual answers**.
- With RAG, I don’t need the most expensive model; I need **faithful, concise** outputs.
- `temperature=0` (or low) reduces hallucinations and keeps tone clinical.

**Why not a bigger or local model?**
- Larger models are overkill/costly for short RAG answers.
- Local LLMs are possible, but infra/quality trade-offs are bigger for a time-boxed challenge.

---

### 8) **Fact Memory** (tiny distilled facts from prior answers)

**Why**
- Users ask meta-questions (“What did I ask first?” “Summarize so far.”).
- I extract 1–3 **short facts** from the assistant’s replies and store them in a small vector index.
- Improves follow-ups without polluting the clinical record.

**Why not just chat history?**
- Chat history is sequential text; **semantic lookup** over distilled facts is faster and more robust when the user references earlier content loosely.

---

### 9) Guardrails: **grounded answers + citations + “not stated” fallback**

**Why**
- Clinical QA must be precise. I:
  - Include only retrieved snippets in the prompt,
  - Ask the model to **answer strictly from context**,
  - Return **citations** (transcript/soap),
  - Say **“Not stated…”** when evidence is insufficient.

**Why not let the model infer?**
- In medical settings, hallucinations are **unacceptable**. Guardrails enforce trust.

---

### 10) Caching & cost control

- Redis naturally caches conversation state.
- Using a **small model** + **targeted context** keeps tokens low.
- Chroma runs locally → **no vector DB bills**.
- Optional: add response caching by normalized question+context to cut LLM calls further.

---

### 11) Scalability story (pragmatic)

- **Stateless API** behind a load balancer; sessions go to **Redis** (shared).
- Vector store can be **replicated** or moved to a managed vector DB if traffic grows.
- Model provider already scales; if moving to on-prem, we can swap the LLM client.
- Frontend on Vercel scales independently; CORS restricts access to the allowed origin.

---

## TL;DR — Why these choices are a **good fit** for the challenge

- **Reliable**: grounded RAG with citations + “not stated” fallback.
- **Responsive**: Redis memory + ensemble retrieval keeps answers quick & relevant.
- **Handles long docs**: chunking + vector search; no need to prompt the whole transcript.
- **Cost-aware**: small model, local Chroma, minimal tokens.
- **Easy to extend**: pluggable models, embeddings, and stores; clean FastAPI endpoints.
- **Deployable**: monorepo, simple envs, Vercel (frontend) + Render (backend) or your infra.

## 🧠 How It Works

* **Embedding:** Transcript + SOAP notes are embedded into Chroma.
* **Retrieval:** The query retrieves relevant chunks via dense + sparse (BM25) search.
* **Generation:** Context is fed to an OpenAI model via LangChain for grounded responses.
* **Display:** Next.js UI shows the answer with citations; Redis stores session history.

## 🧩 Handling Long / Complex Transcripts

* Hierarchical retrieval (coarse section → fine chunk)
* Hybrid dense + sparse retrieval for coverage of medical terms
* Per-session Chroma memory for fact caching
* Conversational context rewrite via LangChain chains

## 💡 Edge Case Handling

| Edge Case | Strategy |
|---|---|
| Incomplete transcripts | Returns *Not stated in transcript/SOAP.* |
| Ambiguous queries | Suggests disambiguation or clarifying question |
| Conflicting data | Combines evidence from multiple sections |
| Repeat questions | Uses cached last assistant response |



### 1) Making follow-up queries more responsive & contextually relevant
**What I do now**
- **Session memory (Redis):** I persist `chat_history` per `session_id` so follow-ups (“…and what about triggers?”) are grounded in prior turns.
- **Question re-write:** LangChain’s conversational chain converts follow-ups into **standalone questions** for retrieval.
- **Semantic chat recall:** I index prior user/assistant turns into a **chat Chroma store** and retrieve them alongside clinical docs.
- **Distilled fact memory:** After each answer, we extract 1–3 short, stable **facts** (duration, meds, tests) and index them for fast follow-ups.


---

### 2) Handling really long recordings / large transcripts
**What I do now**
- **Chunking:** RecursiveCharacterTextSplitter (≈1200/200) to create overlap for continuity.
- **Hybrid retrieval:** Dense embeddings + **BM25** (lexical) ensemble; great for medical terms/drugs.
- **Section-aware docs:** SOAP sections (S/O/A/P) stored as **separate documents** to target clinical facts directly.
- **Answerability check:** If evidence is thin or conflicting, I return **“Not stated in transcript/SOAP or prior chat.”**



---

### 3) Edge cases: incomplete transcripts or ambiguous questions
**What I do now**
- **Ambiguity handling:** if multiple plausible matches, I prefer concise answers; when unclear, I preserve uncertainty and/or cite multiple snippets.
- **Explicit fallback:** “**Not stated in the transcript/SOAP or prior chat.**” rather than hallucinating.
- **History & repeat:** quick paths for “what did I ask first?”, “repeat that”, “summarize so far.”



---


## ✅ Example Queries

* "How long have the symptoms been going on?"
* "What tests were ordered?"
* "Did the patient describe any triggers?"
* "What medications are being used?"
* "What was my first question?" (demonstrates Redis memory)

## ⚪ Evaluation Criteria Mapping

| Criterion | Implementation |
|---|---|
| **Functionality** | Grounded answers with citations from transcript + SOAP. |
| **Integration** | Seamless LLM + Chroma + Redis + FastAPI integration. |
| **Code Quality** | Modularized backend, commented, easily extendable. |
| **User Experience** | Clean Next.js UI, rotating tips, citation toggles. |
| **Innovation** | Hybrid retrieval, per-session memory, multi-layer RAG. |

## 🧪 My Evaluation Process (in simple words)

### 🍀 What I wanted to check

I wanted to see how accurate my MedQuery Chat system really is when a provider asks it questions about the patient transcript and SOAP note. Basically, I asked myself:

> "If I give my bot a set of clinical questions, how close are its answers to the real, correct ones — and are they supported by the actual notes?"

### 🔴 Step 1: I built a "Golden Source"

I created a file called `golden_source.json` with 18 questions and their perfect answers (what a doctor *should* answer based on the transcript). Each question also had a type label:

* **factual** - asking for clear facts
    * *Example: "What tests were ordered?"*
* **context** - about background info
    * *Example: "Where does the patient work?"*
* **instructional** - about the provider's plan or recommendations
    * *Example: "What follow-up was planned?"*

This became my answer key for the evaluation.

### ⚙️ Step 2: I wrote an automatic evaluation script

I wrote a Python script (`evaluate.py`) that goes through every question in that file and:

1.  Sends the question to my running backend (`/ask` endpoint).
2.  Gets back my bot's answer and its citations (the evidence chunks it used).
3.  Compares my bot's answer to the gold one word by word.

Because the answers are short text, not numbers or yes/no, I didn't check exact matches — instead, I looked at how many words overlapped.

---
### 📊 Step 3: How I measured performance

For each answer, I computed:

* **Precision** -> Of all the words my bot said, how many were actually correct?
* **Recall** -> Of all the correct words, how many did my bot find?
* **F1** -> The balance between being correct and being complete.
* **Support** -> Whether the bot's answer was backed by real citations from the transcript or SOAP note.

#### Example

**Gold answer:** "Spirometry, CBC, IgE, allergy testing."
**Bot answer:** "Spirometry and CBC were ordered."

-> **Common words:** spirometry, cbc

Precision = 2 / 3 = 0.67  (2 correct out of 3 words)

Recall = 2 / 4 = 0.5     (found half the gold facts)

F1 = 2 * (0.67 * 0.5) / (0.67 + 0.5) = 0.57

So the bot gets an **F1 of 0.57** for this question — partly right, but incomplete.

### 🏁 Step 4: The output I got

After running all 18 questions, here's what I got:
== Summary ==

Items: 18 | OK: 18 | Errors: 0

Precision: 0.47 | Recall: 0.69 | F1: 0.55 | Support: 0.44 | Avg Latency: 5693 ms
By type:
- factual       Precision 0.45  Recall 0.67  F1 0.53  Support 0.46  n=13
- context       Precision 0.55  Recall 0.71  F1 0.62  Support 0.33  n=3
- instructional Precision 0.48  Recall 0.79  F1 0.60  Support 0.50  n=2

### 🧠 How I interpret these results

#### 🐽 Overall performance

* **Precision = 0.47** -> About half of what my bot said was perfectly correct. It sometimes adds small extras like *"and possibly other tests."*
* **Recall = 0.69** -> It found around 70% of the real facts. It usually gets the right idea, but misses one or two details per question.
* **F1 = 0.55** -> So, on average, my answers are about halfway between perfect and incomplete – not bad for an end-to-end RAG system.
* **Support = 0.44** -> Less than half of the answers had strong citation overlap. That means the model was right sometimes, but *didn't always use the correct evidence*.
* **Latency = 5.7 seconds** -> Each answer takes about 5-6 seconds to generate — decent for a demo; can improve with caching.

### 🔬 By question type

| Type | Precision | Recall | F1 | What it means |
| :--- | :---: | :---: | :---: | :--- |
| **Factual (13 questions)** | 0.45 | 0.67 | 0.53 | It usually finds the correct data (like meds, tests, diagnosis), but adds fluff or misses details — e.g., says "blood test" instead of "CBC with eosinophils." |
| **Context (3 questions)** | 0.55 | 0.71 | 0.62 | It does quite well with background info like "teacher," "lives with roommate," but grounding (Support=0.33) was weaker because these details appear only once in the transcript. |
| **Instructional (2 questions)** | 0.48 | 0.79 | 0.60 | The bot captured provider instructions well (follow-up in 2 weeks, when to go to urgent care), but sometimes over-explained — lowering precision. |

---

### 💬 What these numbers tell me

Overall, the bot:

* Remembers the right information most of the time (good recall).
* Speaks truthfully but wordy (moderate precision).