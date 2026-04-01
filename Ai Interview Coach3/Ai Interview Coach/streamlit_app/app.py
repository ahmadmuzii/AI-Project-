import os

import requests
import streamlit as st

import api_client as api

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎙️",
    layout="wide",
)

FEEDBACK_BOX = """
<div style="background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724;
padding: 14px 16px; border-radius: 10px; margin: 8px 0;">
<strong>💬 Feedback</strong><br/><br/>
{content}
</div>
"""

TRANSCRIPT_BOX = """
<div style="background-color: #f8f9fa; border-left: 4px solid #495057;
padding: 14px 16px; border-radius: 6px; margin: 8px 0;">
<strong>📝 Transcript</strong><br/><br/>
{content}
</div>
"""

WORD_BOX = """
<div style="background-color: #fff3cd; border: 1px solid #ffc107; color: #856404;
padding: 14px 16px; border-radius: 10px; margin: 8px 0;">
<strong>🔍 Word-level analysis</strong>
{content}
</div>
"""


def _init_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "sessions" not in st.session_state:
        st.session_state.sessions = []


def _refresh_sessions() -> None:
    uid = st.session_state.user_id
    if uid is None:
        st.session_state.sessions = []
        return
    try:
        st.session_state.sessions = api.list_sessions(uid)
    except requests.RequestException as e:
        st.error(f"Could not load sessions: {e}")
        st.session_state.sessions = []


def _ensure_session_selected() -> None:
    if st.session_state.current_session_id is None and st.session_state.sessions:
        st.session_state.current_session_id = st.session_state.sessions[0]["id"]


def _play_audio(recording_id: int) -> None:
    url = api.recording_audio_url(recording_id)
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        fmt = "audio/wav"
        if "mpeg" in ctype or "mp3" in ctype:
            fmt = "audio/mpeg"
        elif "mp4" in ctype or "m4a" in ctype:
            fmt = "audio/mp4"
        elif "webm" in ctype:
            fmt = "audio/webm"
        st.audio(r.content, format=fmt)
    except requests.RequestException:
        st.warning("Could not load audio from the server.")


def _render_word_analysis(items: list[dict]) -> None:
    if not items:
        st.markdown(
            WORD_BOX.format(content="<p>No word-level issues flagged.</p>"),
            unsafe_allow_html=True,
        )
        return
    rows = []
    for w in items:
        rows.append(
            f"<tr><td><code>{w.get('timestamp', '')}</code></td>"
            f"<td><strong>{w.get('word', '')}</strong></td>"
            f"<td>{w.get('issue', '')}</td>"
            f"<td>{w.get('suggestion', '')}</td></tr>"
        )
    inner = (
        "<table style='width:100%; border-collapse: collapse; font-size: 0.9rem;'>"
        "<thead><tr><th>Time</th><th>Word</th><th>Issue</th><th>Suggestion</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    st.markdown(WORD_BOX.format(content=inner), unsafe_allow_html=True)


def _render_recordings(session_id: int) -> None:
    try:
        payload = api.list_recordings(session_id)
    except requests.RequestException as e:
        st.error(f"Could not load recordings: {e}")
        return

    recordings = payload.get("recordings") or []
    if not recordings:
        st.info("No recordings in this session yet. Upload or record audio below and click **Analyze**.")
        return

    for rec in recordings:
        rid = rec["id"]
        with st.chat_message("assistant"):
            st.markdown(f"**Recording #{rid}** · _{rec.get('created_at', '')}_")
            _play_audio(rid)
            esc_t = (rec.get("transcript") or "").replace("<", "&lt;")
            st.markdown(TRANSCRIPT_BOX.format(content=esc_t), unsafe_allow_html=True)
            esc_f = (rec.get("feedback") or "").replace("<", "&lt;").replace("\n", "<br/>")
            st.markdown(FEEDBACK_BOX.format(content=esc_f), unsafe_allow_html=True)
            _render_word_analysis(rec.get("word_analysis") or [])


def main() -> None:
    _init_state()
    api.DEFAULT_API = os.environ.get("INTERVIEW_API_URL", "http://127.0.0.1:8000")

    with st.sidebar:
        st.title("🎙️ Interview Coach")
        st.caption(f"API: `{api.DEFAULT_API}`")

        if st.session_state.user_id is None:
            st.subheader("Sign in")
            name = st.text_input("Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            if st.button("Continue", type="primary"):
                if not name.strip() or not email.strip():
                    st.warning("Please enter name and email.")
                else:
                    try:
                        out = api.create_user(name.strip(), email.strip())
                        st.session_state.user_id = out["user_id"]
                        st.session_state.user_name = out["name"]
                        st.session_state.user_email = out["email"]
                        _refresh_sessions()
                        if not st.session_state.sessions:
                            sess = api.start_session(st.session_state.user_id)
                            st.session_state.current_session_id = sess["session_id"]
                            _refresh_sessions()
                        _ensure_session_selected()
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Registration failed: {e}")
            st.stop()

        st.success(f"**{st.session_state.user_name}**  \n{st.session_state.user_email}")

        if st.button("➕ New Session", use_container_width=True):
            try:
                sess = api.start_session(st.session_state.user_id)
                st.session_state.current_session_id = sess["session_id"]
                _refresh_sessions()
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Could not start session: {e}")

        _refresh_sessions()
        _ensure_session_selected()

        if st.session_state.sessions:
            labels = {
                s["id"]: f"Session {s['id']} — {s.get('started_at', '')[:19]}"
                for s in st.session_state.sessions
            }
            ids = [s["id"] for s in st.session_state.sessions]
            current = st.session_state.current_session_id
            if current not in ids:
                current = ids[0]
                st.session_state.current_session_id = current
            idx = ids.index(current)
            choice = st.radio(
                "History",
                options=ids,
                format_func=lambda i: labels[i],
                index=idx,
                key="session_radio",
            )
            st.session_state.current_session_id = choice
        else:
            st.info("No sessions yet. Create one with **New Session**.")

        if st.button("Refresh list", use_container_width=True):
            _refresh_sessions()
            st.rerun()

    st.title("Practice session")
    sid = st.session_state.current_session_id
    if sid is None:
        st.warning("Select or create a session from the sidebar.")
        return

    st.caption(f"Active session ID: **{sid}**")
    _render_recordings(sid)

    st.divider()
    st.subheader("New answer")
    uploaded = st.file_uploader(
        "Upload audio (WAV, MP3, M4A, …)",
        type=["wav", "mp3", "m4a", "webm", "ogg"],
    )
    recorded = None
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Or record here", key="mic")

    analyze = st.button("Analyze", type="primary", use_container_width=True)

    if analyze:
        data = None
        filename = "answer.wav"
        if recorded is not None:
            data = recorded.getvalue()
            filename = "recording.wav"
        elif uploaded is not None:
            data = uploaded.getvalue()
            filename = uploaded.name or "upload.wav"
        else:
            st.warning("Upload a file or use the microphone first.")

        if data:
            with st.spinner("Transcribing and analyzing… (this can take a while)"):
                try:
                    api.upload_audio(
                        st.session_state.user_id,
                        sid,
                        data,
                        filename,
                    )
                    st.success("Analysis complete.")
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
