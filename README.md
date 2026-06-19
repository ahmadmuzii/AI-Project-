# 🎙️ AI Interview Coach — Your Personal AI Interview Wingman

**Stop winging it. Start winning it.**

A full-stack, AI-powered interview prep platform that listens, watches, and teaches you to crush every interview — with real-time speech analysis, webcam body language tracking, LLM-generated coaching, and adaptive mock interviews.

---

# 🖥️ Preview

**AI Interview Coach**

---

# ✨ What It Does

| Feature                   | What It Means For You                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 🧠 Guided Mock Interviews | AI dynamically generates questions tailored to your target role, company, and experience level — no two sessions are the same |
| 🎤 Live Practice Mode     | Record your answer, get instant transcription + scoring + coaching in seconds                                                 |
| 📊 Deep Speech Metrics    | WPM, filler words (um/uh), long pauses, pitch variation, voice energy, hedging language — the works                           |
| 👁️ Webcam Body Language  | MediaPipe tracks your eye contact, head movement, and hand gestures in real time                                              |
| 🤖 Dual-LLM Coaching      | Groq (LLaMA 3.3 70B) for speed + Grok (xAI) for depth — fallback chain never leaves you hanging                               |
| 📄 Resume-Aware AI        | Upload your CV and the AI reads it, then tailors questions & feedback to your actual background                               |
| 🏢 Company Intelligence   | 10+ company profiles (Google, Meta, Amazon, Apple, Stripe, Netflix, Microsoft, etc.) with authentic interview styles          |
| 🔊 AI Voice Interviewer   | ElevenLabs text-to-speech — the AI speaks questions in natural, configurable voices                                           |
| ⌨️ Text or Voice Mode     | Type your answers or speak them — Whisper transcribes either way                                                              |
| 📈 Progress Dashboard     | Score trends, readiness prediction, leaderboard, study plan — track your climb                                                |

---

# 🧠 AI Modules — Under the Hood

## 🎤 Speech Analysis (`analysis_service.py`)

Audio goes through a 4-stage pipeline using **librosa + NumPy**:

### Metrics Analyzed

* Speaking Rate
* Long Pauses
* Filler Words
* Repetitions
* Hedging
* Type-Token Ratio (TTR)
* Pitch Variation
* Jitter / Shimmer

---

## 👁️ Webcam Analysis (`webcam_service.py`)

MediaPipe Face Mesh + Hands running in real time:

* Eye contact score — gaze direction + blink analysis
* Movement score — head stability tracking across frames
* Gesture detection — hand visibility as a confidence signal
* Stress composite — combines eye, movement, and voice energy into a 0–1 stress score

---

## 🤖 LLM Pipeline (`intelligence_service.py`)

```text
User Answer
      ↓
   Whisper
      ↓
Feature Extraction
      ↓
    Scoring
      ↓
 Groq / Grok LLM
      ↓
   Feedback
```

↕︎

```text
analyze_answer_nlp()
(rule-based NLP)
```

### Responsibilities

* **Groq (LLaMA)** — interview questions, greetings, summaries
* **Grok (xAI)** — answer scoring, follow-up generation
* **Fallback Chain** — Grok → Groq → Rule-Based
* **Content Scoring**

  * Relevance
  * Content Quality
  * STAR Usage

Each scored between **0–1**.

---

# 🗂️ Project Anatomy

```text
Ai Interview Coach/
│
├── backend/                          # 🐍 Python FastAPI (port 8000)
│   ├── app/
│   │   ├── main.py                   # FastAPI entry + Whisper model loader
│   │   ├── config.py                 # .env → os.environ
│   │   ├── database.py               # SQLAlchemy + SQLite
│   │   ├── models.py                 # 8 tables: User, Session, Recording, Metrics, QA, Resume, etc.
│   │   ├── schemas.py                # Pydantic validation
│   │   │
│   │   ├── routes/
│   │   │   ├── auth.py
│   │   │   ├── interview.py
│   │   │   ├── audio.py
│   │   │   ├── analytics.py
│   │   │   ├── resume.py
│   │   │   ├── guided_interview.py
│   │   │   └── elevenlabs.py
│   │   │
│   │   ├── services/
│   │   │   ├── intelligence_service.py
│   │   │   ├── analysis_service.py
│   │   │   ├── audio_service.py
│   │   │   ├── webcam_service.py
│   │   │   ├── resume_service.py
│   │   │   ├── company_service.py
│   │   │   └── elevenlabs_service.py
│   │   │
│   │   ├── utils/
│   │   │   ├── pdf_extractor.py
│   │   │   └── rate_limiter.py
│   │   │
│   │   └── data/
│   │       └── swe_questions.csv
│   │
│   ├── uploads/
│   └── run.py
│
└── frontend/                         # ⚛️ React + Vite (port 3000)
    └── src/
        ├── App.jsx
        ├── pages/
        │   ├── Landing.jsx
        │   ├── Login.jsx
        │   ├── Dashboard.jsx
        │   ├── Practice.jsx
        │   ├── GuidedInterview.jsx
        │   ├── AITools.jsx
        │   ├── HistoryPage.jsx
        │   ├── Profile.jsx
        │   └── SettingsPage.jsx
        │
        ├── components/
        │   ├── GlobalEffects.jsx
        │   ├── ParticleField.jsx
        │   ├── GuidedSetup.jsx
        │   ├── GuidedSession.jsx
        │   ├── GuidedSummary.jsx
        │   ├── LiveInterviewSession.jsx
        │   ├── WebcamOverlay.jsx
        │   ├── AiAvatar.jsx
        │   ├── AudioRecorder.jsx
        │   ├── ScoreGauge.jsx
        │   ├── WordAnalysis.jsx
        │   ├── ResumeAnalyzer.jsx
        │   ├── MovementSuggestions.jsx
        │   └── ui/
        │
        ├── context/
        │   ├── AuthContext.jsx
        │   └── ThemeContext.jsx
        │
        ├── hooks/
        │   ├── useTypewriter.js
        │   └── useMagnetic.js
        │
        └── api/client.js
```

---

# 🛠️ Tech Stack

| Layer        | What We Used                      |
| ------------ | --------------------------------- |
| ⚛️ Frontend  | React 18 + Vite + Framer Motion   |
| 🐍 Backend   | Python 3.10+ / FastAPI / Uvicorn  |
| 🗄️ Database | SQLite + SQLAlchemy 2.0           |
| 🎙️ STT      | OpenAI Whisper (tiny)             |
| 🔊 Audio     | Librosa + SciPy                   |
| 👁️ Vision   | OpenCV + MediaPipe                |
| 🤖 LLM       | Groq (LLaMA 3.3 70B) + Grok (xAI) |
| 🗣️ TTS      | ElevenLabs API                    |
| 🔐 Auth      | JWT (python-jose) + bcrypt        |
| ✨ Effects    | WebGL2 custom particle system     |
| 🎨 Styling   | Tailwind CSS + shadcn/ui          |
| 📄 PDF       | pdfplumber → PyPDF2 → pdfminer    |

---

# ⚡ Quick Start

## Prerequisites

* Python 3.10+
* Node.js 18+
* API Keys:

  * Groq
  * xAI (optional)
  * ElevenLabs (optional)

---

## 🖥️ Backend

```bash
cd "Ai Interview Coach/backend"

pip install -r requirements.txt

cp .env.example .env

# Edit .env with your API keys

python run.py
```

```text
http://localhost:8000
```

---

## 🌐 Frontend

```bash
cd "Ai Interview Coach/frontend"

npm install

npm run dev
```

```text
http://localhost:3000
```

---

# 🔑 Environment Variables

| Variable           |
| ------------------ |
| GROQ_API_KEY       |
| XAI_API_KEY        |
| ELEVENLABS_API_KEY |
| JWT_SECRET_KEY     |

---

# 🧩 Architecture Highlights

## 🎯 Dual-LLM Fallback Chain

```text
Grok (xAI) available?
        │
       Yes
        │
    Use Grok
        │
       No
        ↓
Groq (LLaMA) available?
        │
       Yes
        │
    Use Groq
        │
       No
        ↓
 Rule-Based Fallback
```

Every AI function follows this chain, ensuring the application remains operational even when external APIs are unavailable.

---

## 🌀 WebGL2 Particle System

The background particle field uses **curl noise (divergence-free flow fields)** rendered via **WebGL2**, eliminating the Canvas2D tile-seam issue observed in Chrome's GPU compositor.

### Specifications

* 3000 particles
* 3 depth layers
* Cursor swirl interaction
* Real-time GPU rendering

---

# 📊 Scoring Formula

```text
Fluency
= 1 - (0.3 × pauseRate
     + 0.3 × repetitionRate
     + 0.2 × fillerRatio
     + 0.2 × avgPause)

Confidence
= 1 - (0.3 × hedgeRatio
     + 0.2 × pitchVar
     + 0.2 × jitter
     + 0.3 × fillerRatio)

Composure
= 1 - (0.4 × energyVar
     + 0.3 × longPauseRate
     + 0.3 × pitchStd)

Overall
= 0.40 × Fluency
+ 0.35 × Confidence
+ 0.25 × Composure
```

---

# 🏢 Company Profiles

10 hand-crafted company interview profiles:

* Google
* Meta
* Amazon
* Apple
* Netflix
* Stripe
* Microsoft
* McKinsey
* Goldman Sachs
* Deloitte

### Each Profile Includes

* Authentic interview formats
* Coding round patterns
* System design expectations
* Behavioral interview styles
* Known question banks
* Company-specific tone and focus prompts

---

# 👨‍💻 Built By

* M Hannan Najeeb
* Muhammad Ahmad
* Ameer Hamza

---

# 📜 License

**MIT License**

Use it. Break it. Improve it.

**One interview at a time. 🚀**
