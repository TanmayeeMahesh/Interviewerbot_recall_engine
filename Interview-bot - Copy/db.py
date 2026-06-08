"""
db.py — Supabase persistence for the interview bot (Size 1: save-as-you-go, one interview at a time).
All functions fail SAFE: if Supabase is unreachable, they log and return None so the live
interview is never interrupted by a DB problem. The local JSON transcript remains the backup.
"""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Lazy client so a missing/broken config doesn't crash import
_client = None
def _db():
    global _client
    if _client is None:
        try:
            from supabase import create_client
            _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
            print("🗄️  Supabase client ready")
        except Exception as e:
            print(f"❌ Supabase init failed (interview will still run): {e}")
            _client = False          # mark as failed so we don't retry every call
    return _client or None


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_session(bot_id: str, total_questions: int,
                   candidate_name: str = None, candidate_email: str = None,
                   role: str = None) -> str | None:
    """
    Create a candidate (minimal for now) + a session row. Returns session_id (uuid str) or None.
    candidate_* are optional — the JD/resume team will populate candidates fully later.
    """
    db = _db()
    if not db:
        return None
    try:
        cand = db.table("candidates").insert({
            "name": candidate_name, "email": candidate_email, "role": role,
        }).execute()
        candidate_id = cand.data[0]["id"]
        sess = db.table("sessions").insert({
            "candidate_id": candidate_id, "bot_id": bot_id,
            "status": "in_progress", "total_questions": total_questions,
            "started_at": _now(),
        }).execute()
        sid = sess.data[0]["id"]
        print(f"🗄️  session created → {sid}")
        return sid
    except Exception as e:
        print(f"❌ create_session() failed (continuing without DB): {e}")
        return None


def insert_answer(session_id: str, q_id: str, role: str, speaker: str,
                  topic: str, text: str, category: str = None) -> None:
    """Insert one transcript row (question / answer / followup_* / intro / closing)."""
    db = _db()
    if not db or not session_id:
        return
    try:
        db.table("answers").insert({
            "session_id": session_id, "q_id": q_id, "role": role,
            "speaker": speaker, "topic": topic, "text": text,
            "category": category, "created_at": _now(),
        }).execute()
    except Exception as e:
        print(f"❌ insert_answer() failed (continuing): {e}")


def close_session(session_id: str, status: str, questions_reached: int) -> None:
    """Mark the session finished with its completion status."""
    db = _db()
    if not db or not session_id:
        return
    try:
        db.table("sessions").update({
            "status": status, "questions_reached": questions_reached,
            "ended_at": _now(),
        }).eq("id", session_id).execute()
        print(f"🗄️  session closed → {status}")
    except Exception as e:
        print(f"❌ close_session() failed: {e}")


def read_answers(session_id: str) -> list:
    """Read all answer rows for a session, ordered by time (for the final evaluation)."""
    db = _db()
    if not db or not session_id:
        return []
    try:
        res = (db.table("answers").select("*")
               .eq("session_id", session_id)
               .order("created_at").execute())
        return res.data or []
    except Exception as e:
        print(f"❌ read_answers() failed: {e}")
        return []


def get_candidate_for_session(session_id: str) -> dict:
    """Fetch candidate name/email/role linked to a session (for the report header)."""
    db = _db()
    if not db or not session_id:
        return {}
    try:
        sess = db.table("sessions").select("candidate_id").eq("id", session_id).execute()
        if not sess.data:
            return {}
        cid = sess.data[0]["candidate_id"]
        cand = db.table("candidates").select("*").eq("id", cid).execute()
        return cand.data[0] if cand.data else {}
    except Exception as e:
        print(f"❌ get_candidate_for_session() failed: {e}")
        return {}


def save_report(session_id: str, report: dict) -> None:
    """Upsert the final report row (one per session)."""
    db = _db()
    if not db or not session_id:
        return
    try:
        row = {"session_id": session_id, **report}
        db.table("reports").upsert(row, on_conflict="session_id").execute()
        print(f"🗄️  report saved for session {session_id}")
    except Exception as e:
        print(f"❌ save_report() failed: {e}")


def save_recording_url(session_id: str, url: str) -> None:
    """Store the Recall MP4 recording URL on the session (for the HR dashboard)."""
    db = _db()
    if not db or not session_id or not url:
        return
    try:
        db.table("sessions").update({"recording_url": url}).eq("id", session_id).execute()
        print(f"🗄️  recording_url saved for session {session_id}")
    except Exception as e:
        print(f"❌ save_recording_url() failed: {e}")