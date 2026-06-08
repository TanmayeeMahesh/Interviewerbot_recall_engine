import base64, io, json, requests, time, threading
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks
from gtts import gTTS
from groq import Groq
import uvicorn, os
from dotenv import load_dotenv
import db          # Supabase persistence (fail-safe — interview runs even if DB is down)
import evaluator   # final scoring + report (US-AG-07/08)

load_dotenv()
app = FastAPI(title="AI Interview Bot")

# ─── CONFIG ───────────────────────────────────────────────
RECALLAI_API_KEY = os.getenv("RECALLAI_API_KEY")
NGROK_URL        = os.getenv("NGROK_URL")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

BOT_NAME    = "AI Interviewer (Sandbox)"
RECALL_BASE = "https://ap-northeast-1.recall.ai/api/v1"
GATE_MODEL  = "llama-3.1-8b-instant"
REPLY_MODEL = "llama-3.1-8b-instant"

FOLLOWUPS_PER_Q    = 1        # max follow-ups per question
SILENCE_GATE       = 2.0      # silence (s) before checking turn completeness
MAX_TURN_WAIT      = 12.0     # cap waiting for one turn to complete
NO_SHOW_TIMEOUT    = 300      # 5 min to get consent before bot leaves
SILENCE_END_SEC    = 180      # 3 min mid-interview silence → graceful incomplete end
MAX_INTERVIEW_SEC  = 45 * 60  # 45-min safety cap
TTS_WPS            = 2.6      # est. spoken words/sec (speech serialization)

groq_client = Groq(api_key=GROQ_API_KEY)

with open("speech_interview_logic/sample_questions.json") as f:
    QUESTIONS = json.load(f)

state = {
    "bot_id": None, "session_id": None, "question_index": 0,
    "followup_count": 0, "awaiting_followup": False,
    "interview_started": False, "interview_over": False,
    "completion_status": "completed",
    "transcript": [], "covered_concepts": [],
    "answer_buffer": [], "confirming_completion": False,
    "bot_speaking": False, "last_asked": "",
    "join_time": 0.0, "turn_start_time": 0.0,
    "interview_start_time": 0.0, "last_activity": 0.0,
    "silence_timer": None,
}

process_lock = threading.Lock()
speak_lock   = threading.Lock()

def recall_headers():
    return {"Authorization": f"Token {RECALLAI_API_KEY}", "Content-Type": "application/json"}

# ─── SPEAK (serialized + blocking so audio never overlaps) ─
def speak(bot_id, text):
    with speak_lock:
        state["bot_speaking"] = True
        try:
            buf = io.BytesIO()
            gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
            b64 = base64.b64encode(buf.getvalue()).decode()
            r = requests.post(f"{RECALL_BASE}/bot/{bot_id}/output_audio/",
                              headers=recall_headers(), json={"kind": "mp3", "b64_data": b64})
            print(f"📤 output_audio → {r.status_code}")
            if r.status_code not in (200, 201):
                print(f"❌ speak() failed: {r.status_code} — {r.text[:150]}")
                return
            est = max(2.0, len(text.split()) / TTS_WPS + 0.8)
            time.sleep(est)
        except Exception as e:
            print(f"❌ speak() exception: {e}")
        finally:
            state["bot_speaking"] = False
            state["last_activity"] = time.time()
            state["answer_buffer"] = []

def leave_call(bot_id):
    try:
        r = requests.post(f"{RECALL_BASE}/bot/{bot_id}/leave_call/", headers=recall_headers())
        print(f"👋 leave_call → {r.status_code}")
    except Exception as e:
        print(f"❌ leave_call(): {e}")

def get_bot_status(bot_id):
    r = requests.get(f"{RECALL_BASE}/bot/{bot_id}/", headers=recall_headers())
    if r.status_code == 200:
        sc = r.json().get("status_changes", [])
        return sc[-1].get("code", "unknown") if sc else "unknown"
    return "error"

def fetch_and_save_recording(bot_id, session_id):
    """
    After the call ends, retrieve the bot and pull the MP4 download URL from
    media_shortcuts.video_mixed.data.download_url, then save it to the session.
    Bounded retry: the MP4 may take a few seconds to finalize after 'done'.
    NOTE: this S3 URL is temporary (~7 days). Proper fix later = download → Azure Blob.
    """
    if not session_id:
        return
    time.sleep(15)                           # let Recall finish leaving + start stitching the MP4
    for attempt in range(6):                 # ~6 tries over ~30s
        try:
            r = requests.get(f"{RECALL_BASE}/bot/{bot_id}/", headers=recall_headers())
            if r.status_code == 200:
                recordings = r.json().get("recordings", [])
                for rec in recordings:
                    ms = rec.get("media_shortcuts", {}) or {}
                    vm = ms.get("video_mixed", {}) or {}
                    url = (vm.get("data", {}) or {}).get("download_url")
                    if url:
                        print(f"🎥 recording URL retrieved (attempt {attempt+1})")
                        db.save_recording_url(session_id, url)
                        return
            print(f"   🎥 recording not ready yet (attempt {attempt+1}/6)...")
        except Exception as e:
            print(f"❌ fetch_and_save_recording(): {e}")
        time.sleep(5)
    print("⚠️ recording URL not available after retries — fetch later via /bot-status")

def wait_for_join_and_speak(bot_id, intro):
    print(f"⏳ Polling bot {bot_id} for admission...")
    for i in range(45):
        time.sleep(2)
        status = get_bot_status(bot_id)
        print(f"   [{i+1}/45] status = '{status}'")
        if status in ("in_call_recording", "in_call_not_recording"):
            print("🎉 Bot admitted — speaking intro")
            state["join_time"] = time.time()
            threading.Thread(target=no_show_watchdog, args=(bot_id,), daemon=True).start()
            log_to_transcript(BOT_NAME, intro, q_id="intro", role="intro")
            speak(bot_id, intro)
            return
        if status in ("done", "error", "fatal", "call_ended"):
            print(f"❌ Bot ended early: {status}"); return
    print("❌ Timeout — never reached in_call status")

def no_show_watchdog(bot_id):
    deadline = state["join_time"] + NO_SHOW_TIMEOUT
    while time.time() < deadline:
        time.sleep(5)
        if state["interview_started"] or state["bot_id"] != bot_id:
            return
    if not state["interview_started"] and state["bot_id"] == bot_id:
        print("⏰ No consent within timeout — leaving")
        speak(bot_id, "I haven't received a response, so I'll end the session now. Thank you.")
        leave_call(bot_id)

def cap_watchdog(bot_id):
    while state["bot_id"] == bot_id and not state["interview_over"]:
        time.sleep(10)
        if state["interview_start_time"] and (time.time() - state["interview_start_time"]) >= MAX_INTERVIEW_SEC:
            if state["interview_over"]:
                return
            state["completion_status"] = "capped"
            print("⏲️ 45-min cap reached")
            end_session(bot_id,
                "We're at our time limit for today, so I'll wrap up here. Thank you for your time — "
                "your responses will be sent for further assessment and our team will be in touch.")
            return

def silence_end_watchdog(bot_id):
    while state["bot_id"] == bot_id and not state["interview_over"]:
        time.sleep(5)
        if not state["interview_started"] or state["bot_speaking"]:
            continue
        if state["last_activity"] and (time.time() - state["last_activity"]) >= SILENCE_END_SEC:
            if state["interview_over"]:
                return
            state["completion_status"] = "incomplete_no_response"
            print("🔇 No response for too long — closing as incomplete")
            end_session(bot_id,
                "I haven't heard a response for a while, so I'll conclude the session here. "
                "Thank you for your time — what we covered will be sent for further assessment.")
            return

def end_session(bot_id, closing_text):
    if state["interview_over"] and state["completion_status"] == "completed":
        return
    state["interview_over"] = True
    log_to_transcript(BOT_NAME, closing_text, q_id="closing", role="closing")
    speak(bot_id, closing_text)
    save_transcript()
    sid = state.get("session_id")
    db.close_session(sid, state["completion_status"], state["question_index"] + 1)
    leave_call(bot_id)
    # Final evaluation runs in background so it never delays the bot leaving (US-AG-07 AC-05)
    if sid:
        threading.Thread(target=evaluator.evaluate_session,
                         args=(sid, state["completion_status"]), daemon=True).start()
        # Retrieve + store the MP4 recording URL in the background (US-AG-06)
        threading.Thread(target=fetch_and_save_recording, args=(bot_id, sid), daemon=True).start()

def log_to_transcript(speaker, text, category=None, topic=None, q_id=None, role=None):
    entry = {"timestamp": datetime.now().strftime("%H:%M:%S"),
             "q_id": q_id, "role": role, "speaker": speaker, "topic": topic,
             "text": text, "category": category}
    state["transcript"].append(entry)
    c = f" | {category}" if category else ""
    qs = f" [{q_id}/{role}]" if q_id else ""
    print(f"📝{qs} {speaker}: {text[:60]}{c}")
    # also persist to Supabase (fail-safe; no-op until session_id exists)
    db.insert_answer(state.get("session_id"), q_id, role, speaker, topic, text, category)

def save_transcript():
    fn = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "completion_status": state["completion_status"],
        "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions_reached": state["question_index"] + 1,
        "total_questions": len(QUESTIONS),
        "transcript": state["transcript"],
    }
    with open(fn, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Saved → {fn}  (status: {state['completion_status']})")

# ─── SHARED MODALITY PREFIX ───────────────────────────────
IO_CONTEXT = """INPUT NOTE: The candidate's words come from speech-to-text — no punctuation, words may
be dropped or doubled, homophones may appear (e.g. "emb" may mean "MBA"). Judge MEANING charitably;
never penalize transcription noise.
OUTPUT NOTE: If you write a spoken line, it is read aloud by text-to-speech: short sentences, no
symbols, no lists, no markdown. Sound human and warm, not robotic."""

# ─── 3-STATE TURN DETECTION ───────────────────────────────
def turn_verdict(text: str) -> str:
    if len(text.split()) >= 25:
        return "complete"
    if state.get("confirming_completion"):
        return "complete"
    prompt = f"""{IO_CONTEXT}

Is this spoken interview answer COMPLETE, still going (INCOMPLETE), or genuinely UNCERTAIN?

ANSWER SO FAR: "{text}"

Reply ONE word:
COMPLETE   — a finished thought
INCOMPLETE — clearly trailed off mid-sentence ("and i", "firstly i will")
UNCERTAIN  — grammatically complete but likely more to say ("i would use some algorithms")"""
    try:
        resp = groq_client.chat.completions.create(
            model=GATE_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=4)
        v = resp.choices[0].message.content.strip().upper()
        verdict = "incomplete" if "INCOMPLETE" in v else ("uncertain" if "UNCERTAIN" in v else "complete")
        print(f"🧠 turn verdict: '{text[:40]}...' → {verdict}")
        return verdict
    except Exception as e:
        print(f"❌ turn_verdict(): {e} — defaulting complete")
        return "complete"

# ─── LIVE GATE ────────────────────────────────────────────
def gate_answer(question, answer, key_concepts, topic) -> dict:
    prompt = f"""{IO_CONTEXT}

You are gating ONE interview answer to decide whether to follow up. NOT a final grade.

QUESTION: {question}
TOPIC: {topic}
EXPECTED KEY CONCEPTS: {key_concepts}
CANDIDATE ANSWER: {answer}

Judge: COVERAGE (addressed key concepts?), RELEVANCE (on-topic or dodged?), and most importantly
UNDERSTANDING — did they DEMONSTRATE understanding (explain how/why, give specifics) or just NAME
things (say a term with no explanation)? Confident jargon with no real explanation is NAMING.

Pick ONE category:
  "strong"    — relevant, covers key concepts, shows real understanding
  "thin"      — only NAMES concepts, or confident without substance
  "vague"     — too little / unclear to judge
  "off_topic" — didn't answer, dodged, or drifted

Reply ONLY valid JSON, no markdown:
{{"category":"strong|thin|vague|off_topic","covered":["<demonstrated concept>"],"note":"<5-8 words>"}}"""
    try:
        resp = groq_client.chat.completions.create(
            model=GATE_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.2)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        r = json.loads(raw)
        print(f"🎯 gate={r.get('category')} | covered={r.get('covered')} | {r.get('note','')}")
        return r
    except Exception as e:
        print(f"❌ gate_answer(): {e} — defaulting 'vague'")
        return {"category": "vague", "covered": [], "note": "gate failed"}

# ─── SPOKEN LINES ─────────────────────────────────────────
def followup_line(question, answer, category) -> str:
    styles = {
        "thin": ("They sound confident but may only be NAMING concepts. Ask ONE pointed follow-up "
                 "that tests whether they truly understand a specific term or claim they made."),
        "vague": ("Their answer was unclear or thin. Ask ONE follow-up that helps them give a fuller, "
                  "more specific answer."),
        "off_topic": ("They didn't actually answer. Gently redirect them back to the question with ONE "
                      "clear restatement."),
    }
    prompt = f"""{IO_CONTEXT}

QUESTION ASKED: {question}
CANDIDATE SAID: {answer}

{styles.get(category, styles['vague'])}
Reference a SPECIFIC detail they actually said. Reply with ONLY the spoken follow-up."""
    try:
        resp = groq_client.chat.completions.create(
            model=REPLY_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=80, temperature=0.7)
        return (resp.choices[0].message.content.strip().strip('"') or "Could you go a bit deeper on that?")
    except Exception as e:
        print(f"❌ followup_line(): {e}"); return "Could you go a bit deeper on that?"

def rephrase_question(question, key_concepts) -> str:
    prompt = f"""{IO_CONTEXT}
The candidate did not understand this question. Rephrase it more simply and concretely with a tiny
hint of what you're looking for. Do NOT just repeat it.
QUESTION: {question}
LOOKING FOR: {key_concepts}
Reply with ONLY the rephrased spoken question."""
    try:
        resp = groq_client.chat.completions.create(
            model=REPLY_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=90, temperature=0.6)
        return resp.choices[0].message.content.strip().strip('"') or f"Let me put it differently. {question}"
    except Exception as e:
        print(f"❌ rephrase_question(): {e}"); return f"Let me put it differently. {question}"

def transition_line(answer, next_question) -> str:
    """Natural segue into the next question — replaces the mechanical 'Next,' prefix."""
    prompt = f"""{IO_CONTEXT}

The candidate just said: "{answer}"
The next question to ask is: "{next_question}"

Briefly acknowledge something specific they said (one short clause), then naturally lead into asking
the next question. Sound like a human interviewer segueing — do NOT say "Next" or "Question 5".
Reply with ONLY the spoken sentence(s)."""
    try:
        resp = groq_client.chat.completions.create(
            model=REPLY_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=110, temperature=0.7)
        return resp.choices[0].message.content.strip().strip('"') or f"Thank you. {next_question}"
    except Exception as e:
        print(f"❌ transition_line(): {e}"); return f"Thank you. {next_question}"

def ack_line(answer) -> str:
    prompt = f"""{IO_CONTEXT}
The candidate just said: "{answer}"
Give a brief, warm one-line acknowledgment referencing something specific they said, then stop.
Reply with ONLY the spoken sentence."""
    try:
        resp = groq_client.chat.completions.create(
            model=REPLY_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.7)
        return (resp.choices[0].message.content.strip().strip('"') or "Thank you.")
    except Exception as e:
        print(f"❌ ack_line(): {e}"); return "Thank you."

# ─── CROSS-QUESTION CHECK ─────────────────────────────────
def check_if_already_answered(next_q, next_concepts) -> dict:
    if not state["covered_concepts"]:
        return {"already_answered": False}
    prompt = f"""{IO_CONTEXT}

NEXT QUESTION: {next_q}
ITS KEY CONCEPTS: {next_concepts}
CONCEPTS ALREADY DEMONSTRATED: {state["covered_concepts"]}

Has the candidate substantially answered this already?
Reply ONLY valid JSON:
{{"already_answered":false,"acknowledgment":null,"adjusted_question":null}}
OR:
{{"already_answered":true,"acknowledgment":"You touched on this earlier.","adjusted_question":"Building on that, can you go deeper into X?"}}"""
    try:
        resp = groq_client.chat.completions.create(
            model=GATE_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=150)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        r = json.loads(raw)
        if r.get("already_answered"):
            print("⚡ Cross-Q: already covered — adapting")
        return r
    except Exception as e:
        print(f"❌ check_if_already_answered(): {e}")
        return {"already_answered": False}

# ─── TURN ENDING ──────────────────────────────────────────
def schedule_processing(bot_id):
    if state["silence_timer"]:
        state["silence_timer"].cancel()
    t = threading.Timer(SILENCE_GATE, run_gate_check, args=(bot_id,))
    t.daemon = True
    state["silence_timer"] = t
    t.start()

def run_gate_check(bot_id):
    if state["interview_over"]:
        return
    if not process_lock.acquire(blocking=False):
        print("🔒 already processing — skipping duplicate fire")
        return
    try:
        if not state["answer_buffer"] or state["bot_speaking"]:
            return
        full_answer = " ".join(state["answer_buffer"]).strip()
        waited = time.time() - (state.get("turn_start_time") or time.time())
        verdict = turn_verdict(full_answer) if waited < MAX_TURN_WAIT else "complete"

        if verdict == "incomplete":
            print("   ↳ incomplete — keep listening")
            schedule_processing(bot_id); return
        if verdict == "uncertain":
            print("   ↳ uncertain — asking if they're done")
            state["confirming_completion"] = True
            speak(bot_id, "Did you want to add anything, or shall we continue?")
            schedule_processing(bot_id); return

        state["answer_buffer"] = []
        state["turn_start_time"] = 0.0
        state["confirming_completion"] = False
        print(f"🧩 Full answer: '{full_answer}'")
        process_answer(bot_id, full_answer)
    finally:
        process_lock.release()

# ─── CORE FLOW ────────────────────────────────────────────
def process_answer(bot_id, candidate_text):
    if state["interview_over"]:
        return
    low = candidate_text.lower()
    item = QUESTIONS[state["question_index"]]
    q_num = state["question_index"] + 1
    q_now = item["question"]

    state["confirming_completion"] = False

    last = state.get("last_asked") or q_now   # the actual last thing the bot asked
    if any(p in low for p in ["repeat", "say again", "come again", "didn't catch", "didn't hear"]):
        print("🔁 Meta: repeat last-asked"); speak(bot_id, f"Of course. {last}"); return
    if any(p in low for p in ["don't understand", "didn't understand", "not clear", "what do you mean", "confused", "rephrase"]):
        print("💡 Meta: rephrase last-asked"); speak(bot_id, rephrase_question(last, item["key_concepts"])); return
    if any(p in low for p in ["still thinking", "give me a moment", "one moment", "hold on", "let me think"]):
        print("🧠 Meta: thinking"); speak(bot_id, "Take your time. I'm listening whenever you're ready."); return

    gate = gate_answer(item["question"], candidate_text, item["key_concepts"], item["topic"])
    category = gate.get("category", "vague")
    state["covered_concepts"] = list(set(state["covered_concepts"] + gate.get("covered", [])))

    if state["awaiting_followup"]:
        log_to_transcript("Candidate", candidate_text, category=category,
                          topic=item["topic"], q_id=f"{q_num}.1", role="followup_answer")
        advance(bot_id, candidate_text)
        return

    log_to_transcript("Candidate", candidate_text, category=category,
                      topic=item["topic"], q_id=str(q_num), role="answer")

    can_followup = (state["followup_count"] < FOLLOWUPS_PER_Q)
    if category != "strong" and can_followup:
        state["followup_count"] += 1
        state["awaiting_followup"] = True
        fu = followup_line(item["question"], candidate_text, category)
        state["last_asked"] = fu
        print(f"🔄 Follow-up {q_num}.1 ({category})")
        log_to_transcript(BOT_NAME, fu, topic=item["topic"], q_id=f"{q_num}.1", role="followup_question")
        speak(bot_id, fu)
        return

    advance(bot_id, candidate_text)

def advance(bot_id, last_answer):
    if state["interview_over"]:
        return
    state["awaiting_followup"] = False
    state["followup_count"] = 0
    state["question_index"] += 1

    if state["question_index"] < len(QUESTIONS):
        nxt = QUESTIONS[state["question_index"]]
        nxt_num = state["question_index"] + 1
        cc = check_if_already_answered(nxt["question"], nxt["key_concepts"])
        if cc.get("already_answered") and cc.get("adjusted_question"):
            ack = ack_line(last_answer)
            to_say = f"{ack} {cc.get('acknowledgment','')} {cc['adjusted_question']}"
        else:
            to_say = transition_line(last_answer, nxt["question"])
        state["last_asked"] = nxt["question"]   # repeat/rephrase target = the question itself
        log_to_transcript(BOT_NAME, to_say, topic=nxt["topic"], q_id=str(nxt_num), role="question")
        speak(bot_id, to_say)
    else:
        state["completion_status"] = "completed"
        ack = ack_line(last_answer)
        end_session(bot_id, f"{ack} That completes our interview. Thank you for your time — your "
                            "responses will be sent for further assessment and our team will be in touch.")
        print("✅ Interview complete")

# ─── ENDPOINTS ────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running", "q": f"{state['question_index']+1}/{len(QUESTIONS)}",
            "started": state["interview_started"], "over": state["interview_over"],
            "completion_status": state["completion_status"], "session_id": state["session_id"]}

@app.get("/trigger-bot")
def trigger_bot(meeting_url: str, background_tasks: BackgroundTasks):
    print(f"\n🚀 Deploying to: {meeting_url}")
    state.update({"bot_id": None, "session_id": None, "question_index": 0, "followup_count": 0,
                  "awaiting_followup": False, "interview_started": False, "interview_over": False,
                  "completion_status": "completed", "transcript": [], "covered_concepts": [],
                  "answer_buffer": [], "confirming_completion": False, "bot_speaking": False,
                  "last_asked": "",
                  "join_time": 0.0, "turn_start_time": 0.0, "interview_start_time": 0.0,
                  "last_activity": 0.0, "silence_timer": None})
    payload = {
        "bot_name": BOT_NAME, "meeting_url": meeting_url,
        "recording_config": {
            "video_mixed_mp4": {},   # produce a combined audio+video MP4 artifact (US-AG-06)
            "transcript": {"provider": {"recallai_streaming": {
                "mode": "prioritize_low_latency", "language_code": "en"}}},
            "realtime_endpoints": [{"type": "webhook",
                "url": f"{NGROK_URL}/webhook/transcription",
                "events": ["transcript.data", "transcript.partial_data"]}]
        },
        "automatic_leave": {
            "waiting_room_timeout": 600,
            "in_call_not_recording_timeout": 3600,
            "silence_detection": {"timeout": 3600}
        }
    }
    r = requests.post(f"{RECALL_BASE}/bot/", headers=recall_headers(), json=payload)
    if r.status_code == 201:
        bot_id = r.json()["id"]; state["bot_id"] = bot_id
        intro = ("Hello! I'm your AI interviewer today. This session is recorded — both audio and "
                 "video — and transcribed. Do you consent to proceed? Please say yes to continue.")
        background_tasks.add_task(wait_for_join_and_speak, bot_id, intro)
        print(f"✅ Bot deployed: {bot_id}")
        return {"status": "deployed", "bot_id": bot_id}
    print(f"❌ Deploy failed: {r.status_code} — {r.text}")
    return {"status": "failed", "detail": r.text}

@app.get("/test-speak/{bot_id}")
def test_speak(bot_id):
    speak(bot_id, "Hello, this is a test. Can you hear me?")
    return {"status": "attempted — check terminal"}

@app.post("/webhook/transcription")
async def handle_transcription(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    event = payload.get("event", "")
    data  = payload.get("data", {})

    inner    = data.get("data", {}) if isinstance(data, dict) else {}
    words    = inner.get("words", [])
    text     = " ".join(w.get("text", "") for w in words).strip()
    speaker  = (inner.get("participant", {}) or {}).get("name", "")
    is_final = (event == "transcript.data")
    bot_id   = payload.get("bot", {}).get("id") or state.get("bot_id", "")

    if event in ("bot.done", "call.ended"):
        if state["transcript"] and not state["interview_over"]:
            save_transcript()
            db.close_session(state.get("session_id"), state["completion_status"], state["question_index"] + 1)
        # Retrieve + store the MP4 recording URL in the background (may take a few seconds to finalize)
        if state.get("session_id"):
            threading.Thread(target=fetch_and_save_recording,
                             args=(bot_id, state["session_id"]), daemon=True).start()
        return {"status": "ok"}

    if not text or speaker == BOT_NAME or state["interview_over"]:
        return {"status": "ok"}
    if state["bot_speaking"]:
        return {"status": "ok"}

    if not state["interview_started"]:
        if not is_final:
            return {"status": "ok"}
        no_words  = ["no", "don't", "stop", "refuse"]
        yes_words = ["yes", "sure", "okay", "ok", "proceed", "agree", "yeah", "yep", "consent"]
        if any(w in text.lower() for w in no_words):
            background_tasks.add_task(speak, bot_id, "Understood. Interview cancelled. Thank you.")
            background_tasks.add_task(leave_call, bot_id)
        elif any(w in text.lower() for w in yes_words) or len(text.split()) <= 3:
            state["interview_started"] = True
            state["interview_start_time"] = time.time()
            state["last_activity"] = time.time()
            # create the DB session now that consent is given
            state["session_id"] = db.create_session(bot_id, len(QUESTIONS))
            threading.Thread(target=cap_watchdog, args=(bot_id,), daemon=True).start()
            threading.Thread(target=silence_end_watchdog, args=(bot_id,), daemon=True).start()
            first = QUESTIONS[0]
            state["last_asked"] = first["question"]
            log_to_transcript(BOT_NAME, first["question"], topic=first["topic"], q_id="1", role="question")
            background_tasks.add_task(speak, bot_id, f"Wonderful, thank you. Let's begin. {first['question']}")
            print("✅ Consent received — interview starting")
        else:
            background_tasks.add_task(speak, bot_id, "Please say yes to begin.")
        return {"status": "ok"}

    if state["question_index"] < len(QUESTIONS):
        if is_final:
            state["last_activity"] = time.time()
            if not state["answer_buffer"]:
                state["turn_start_time"] = time.time()
            state["answer_buffer"].append(text)
        schedule_processing(bot_id)
    return {"status": "ok"}

@app.get("/transcript")
def get_transcript():
    return {"completion_status": state["completion_status"],
            "session_id": state["session_id"],
            "entries": state["transcript"], "count": len(state["transcript"])}

@app.get("/bot-status/{bot_id}")
def bot_status(bot_id):
    return {"bot_id": bot_id, "status": get_bot_status(bot_id)}

@app.get("/stop-bot/{bot_id}")
def stop_bot(bot_id):
    state["interview_over"] = True
    leave_call(bot_id)
    save_transcript()
    db.close_session(state.get("session_id"), "stopped", state["question_index"] + 1)
    return {"status": "stopped"}

if __name__ == "__main__":
    print("\n" + "="*55)
    print(f"  {len(QUESTIONS)} Qs | category gate | {FOLLOWUPS_PER_Q} follow-up/Q | "
          f"silence-end {SILENCE_END_SEC//60}min | cap {MAX_INTERVIEW_SEC//60}min | DB on")
    print("="*55 + "\n")
    uvicorn.run("app_full:app", host="0.0.0.0", port=8000, reload=True)