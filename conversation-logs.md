# Conversation Logs

## Session 1 — 2026-05-22

### Project Scan Results

**Project:** AI Interview Coach

**Overview:** A full-stack web application for AI-powered mock interview practice with speech analysis, webcam stress detection, resume analysis, and guided interviews.

---

### Project Structure

```
Project (Ai interview coach)/
├── requirements.txt                     # Root-level Python deps
├── conversation-logs.md                 # This file
├── .venv/                               # Python virtual environment
└── Ai Interview Coach/
    ├── backend/
    │   ├── run.py                       # Entry point (empty)
    │   ├── interview_coach.db           # SQLite database
    │   ├── .env                         # API keys (GROQ, ELEVENLABS)
    │   ├── .vscode/settings.json
    │   ├── requirements.txt
    │   ├── uploads/                     # Audio recordings & avatars
    │   └── app/
    │       ├── main.py                  # FastAPI app entry, lifespan, middleware, routing
    │       ├── config.py                # Env loading (GROQ, ELEVENLABS, XAI, GROK keys)
    │       ├── database.py              # SQLAlchemy engine & session (SQLite)
    │       ├── models.py                # ORM models (User, InterviewSession, Recording, etc.)
    │       ├── schemas.py               # Pydantic schemas
    │       ├── routes/
    │       │   ├── auth.py              # Register, login, profile, avatar/resume upload
    │       │   ├── interview.py         # Sessions, recordings CRUD, audio streaming
    │       │   ├── audio.py             # Upload & transcribe audio, NLP + LLM scoring
    │       │   ├── analytics.py         # Dashboard, leaderboard, stress, study plan, etc.
    │       │   ├── resume.py            # Resume upload, ATS scoring, skills gap analysis
    │       │   ├── guided_interview.py  # Guided interview flow (greeting, Q&A, summary)
    │       │   └── elevenlabs.py        # ElevenLabs TTS voices & speech
    │       ├── services/
    │       │   ├── analysis_service.py  # Word-level analysis, temporal/fluency/lexical/acoustic features
    │       │   ├── intelligence_service.py # LLM calls (Grok/Groq), NLP, question gen, summarization
    │       │   ├── audio_service.py     # Audio save & transcribe orchestration
    │       │   ├── resume_service.py    # Resume extraction, ATS scoring, skills gap
    │       │   ├── webcam_service.py    # MediaPipe/OpenCV face tracking, eye contact, movement
    │       │   ├── elevenlabs_service.py # ElevenLabs API: list voices, TTS
    │       │   └── company_service.py   # Company profiles (Google, Meta, Amazon, etc.)
    │       └── utils/
    │           ├── pdf_extractor.py     # PDF text extraction (pdfplumber, PyPDF2, pdfminer)
    │           └── file_handler.py      # (Empty)
    └── frontend/
        ├── package.json                # React 19, Vite 8, framer-motion, axios
        ├── vite.config.js               # Dev server on :3000, proxy /uploads to :8000
        ├── index.html
        ├── eslint.config.js
        ├── README.md
        ├── .gitignore
        ├── dist/                       # Built files
        └── src/
            ├── main.jsx                # App entry, strict mode, theme provider
            ├── App.jsx                 # Client-side routing (react-router-dom v7)
            ├── index.css               # Design system (dark/light theme, CSS variables)
            ├── context/
            │   ├── AuthContext.jsx      # Auth state, login, register, profile
            │   └── ThemeContext.jsx     # Dark/light theme toggle
            ├── api/
            │   └── client.js           # Axios instance with JWT interceptor
            ├── pages/
            │   ├── Landing.jsx         # Landing page
            │   ├── Login.jsx           # Login/Register page
            │   ├── Dashboard.jsx       # User dashboard with stats
            │   ├── Practice.jsx        # Practice session (record, analyze)
            │   ├── GuidedInterview.jsx # Guided interview flow
            │   ├── AITools.jsx         # AI-powered tools hub
            │   ├── Profile.jsx         # User profile management
            │   ├── SettingsPage.jsx    # User settings
            │   └── HistoryPage.jsx     # Session history
            └── components/
                ├── Navbar.jsx / .css
                ├── ChatSession.jsx
                ├── AudioPlayer.jsx
                ├── AudioRecorder.jsx
                ├── AnalysisResults.jsx
                ├── ScoreGauge.jsx
                ├── WordAnalysis.jsx
                ├── MovementSuggestions.jsx
                ├── WebcamOverlay.jsx
                ├── LiveInterviewSession.jsx
                ├── GuidedSetup.jsx
                ├── GuidedSession.jsx
                ├── GuidedSummary.jsx
                ├── AiAvatar.jsx
                ├── ResumeManager.jsx
                ├── ResumeCard.jsx
                ├── ResumeDropzone.jsx
                ├── ResumeAnalyzer.jsx
                ├── ManualResumeForm.jsx
                ├── OnboardingModal.jsx
                ├── ThemeToggle.jsx
                └── (others)
```

---

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python FastAPI + SQLAlchemy + SQLite |
| **Frontend** | React 19 + Vite 8 + React Router v7 |
| **Auth** | JWT (python-jose) + bcrypt |
| **Speech** | OpenAI Whisper (tiny) + librosa |
| **LLMs** | Groq (llama-3.3-70b) + xAI Grok (grok-2) |
| **Vision** | MediaPipe / OpenCV (face mesh, hands) |
| **TTS** | ElevenLabs API |
| **Resume** | PyPDF2 / pdfplumber / pdfminer, ATS scoring |
| **Styling** | CSS custom properties (dark/light), framer-motion |

---

### Features

1. **User Auth** — Register/login with JWT, profile management, avatar upload
2. **Practice Sessions** — Record audio answers, Whisper transcription, word-level analysis (fillers, hedging, pauses), acoustic/prosodic feature extraction
3. **LLM Feedback** — Groq/Grok-powered feedback on content, confidence, fluency
4. **Guided Interviews** — Multi-turn interview with greeting, clarifying questions, adaptive follow-ups, scoring, and summary
5. **Webcam Analysis** — Real-time face tracking, eye contact, movement detection via MediaPipe
6. **Resume Analysis** — PDF upload, skill extraction, ATS scoring, skills gap analysis
7. **Company Mode** — Tailored questions for Google, Meta, Amazon, Apple, Netflix, Stripe, Microsoft, etc.
8. **Analytics Dashboard** — Score trends, heatmaps, streak tracking, readiness prediction, leaderboard
9. **ElevenLabs TTS** — Text-to-speech for AI interviewer voice

### API Routes Summary

| Prefix | Routes |
|--------|--------|
| `/auth` | register, login, me, profile, upload-avatar, upload-resume, change-password, delete-account |
| `/interview` | start-session, sessions, recordings, delete, audio streaming |
| `/upload-audio` | POST/GET upload audio, transcribe, score |
| `/analytics` | dashboard, leaderboard, study-plan, session-summary, adaptive questions, company mode |
| `/resume` | list, upload, manual, set-primary, delete, profile, analysis |
| `/guided` | list, start, answer-clarification, answer, next-question, complete |
| `/elevenlabs` | list-voices, text-to-speech |

### Database Models

- `users` — Full user profile with career context, preferences, auth
- `sessions` — Interview sessions per user
- `recordings` — Audio recordings with transcript & feedback
- `word_analysis` — Per-word issues (filler, hedge, repetition, pause)
- `recording_metrics` — Numerical scores (fluency, confidence, composure, overall, etc.)
- `user_resumes` — Uploaded resumes with extracted data
- `guided_interviews` — Guided interview sessions
- `interview_qa` — Q&A pairs within guided interviews

---

### Key Dependencies

**Backend (Python):** fastapi, uvicorn, sqlalchemy, whisper, librosa, numpy, scipy, torch, groq, opencv-python, mediapipe, python-jose, passlib, bcrypt, httpx, PyPDF2, pdfplumber, python-dotenv, python-multipart

**Frontend (JS):** react, react-dom, react-router-dom, axios, framer-motion, @mediapipe/tasks-vision

### Security Notes
- JWT secret key hardcoded in `auth.py` (`"aic-secret-key-change-in-production-2026"`)
- API keys (GROQ, ELEVENLABS) in `.env` file (not gitignored apparently — `.env` exists in repo)
- CORS wide open (`allow_origins=["*"]`)
