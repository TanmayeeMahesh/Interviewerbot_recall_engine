"""
evaluator.py — Final candidate evaluation (US-AG-07) + recommendation report (US-AG-08).
Runs ONCE at interview end, over the whole transcript read from the DB. Separate from the
live category gate — this is the rigorous, auditable scoring that produces the HR report.

Scoring (US-AG-07 AC-02): each TOPIC scored 1-10 on four dimensions:
  Technical Accuracy 40% | Depth 30% | Clarity 20% | Problem-Solving 10%
Per-topic = weighted avg of those four (AC-03). Composite = avg across topics (AC-04).
"""
import os, json
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import db

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
EVAL_MODEL = "llama-3.3-70b-versatile"     # rigorous judge; runs in background, latency hidden

# Dimension weights (backend-defined, not HR-editable — US-AG-07 AC-01)
WEIGHTS = {"technical_accuracy": 0.40, "depth": 0.30, "clarity": 0.20, "problem_solving": 0.10}


def _group_by_topic(answers: list) -> dict:
    """
    Group transcript rows into {topic: {question, answer, followup_q, followup_a, live_categories}}.
    Uses role to assemble the full exchange per question/topic.
    """
    topics = {}
    for row in answers:
        role = row.get("role")
        topic = row.get("topic") or "General"
        if role in ("intro", "closing") or not topic:
            continue
        t = topics.setdefault(topic, {"question": "", "answer": "", "followup_q": "",
                                      "followup_a": "", "categories": []})
        text = (row.get("text") or "").strip()
        if role == "question":
            t["question"] = text
        elif role == "answer":
            t["answer"] = text
            if row.get("category"): t["categories"].append(row["category"])
        elif role == "followup_question":
            t["followup_q"] = text
        elif role == "followup_answer":
            t["followup_a"] = text
            if row.get("category"): t["categories"].append(row["category"])
    return topics


def _score_topic(topic: str, block: dict) -> dict:
    """Score ONE topic on the four dimensions with the 70B model."""
    live_hint = ", ".join(block["categories"]) or "none"
    prompt = f"""You are a STRICT senior technical interviewer producing a FINAL, auditable score for
one interview topic. Input is speech-to-text (no punctuation, words may be doubled or dropped, "emb"
may mean "MBA") — judge MEANING charitably, never penalize transcription noise.

Be discriminating. Most answers are NOT a 9-10. Reserve high scores for genuine depth and correctness.
Confident jargon with no real explanation scores LOW on accuracy and depth even if it sounds fluent.

TOPIC: {topic}
QUESTION: {block['question']}
CANDIDATE ANSWER: {block['answer']}
FOLLOW-UP ASKED: {block['followup_q'] or '(none)'}
FOLLOW-UP ANSWER: {block['followup_a'] or '(none)'}
LIVE FIRST-PASS SIGNAL (hint only, may be wrong): {live_hint}

Score EACH dimension 1-10:
  technical_accuracy — correct, on-topic, no errors (1 = wrong/absent, 10 = fully correct)
  depth — specifics, examples, trade-offs (1 = vague one-liner, 10 = concrete and thorough)
  clarity — structured and coherent (1 = incoherent, 10 = crystal clear)
  problem_solving — reasoning/approach visible (1 = none, 10 = clear methodology)

Also set "answered": false ONLY if the candidate gave essentially NO content (e.g. just "yes",
"I don't know", silence, or repeated a previous answer without addressing THIS topic). If they made
a genuine attempt with real content — even if weak — set "answered": true.

Reply ONLY valid JSON, no markdown:
{{"technical_accuracy":<1-10>,"depth":<1-10>,"clarity":<1-10>,"problem_solving":<1-10>,
"answered":<true|false>,"note":"<one sentence citing something specific they said>"}}"""
    try:
        resp = groq_client.chat.completions.create(
            model=EVAL_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.2)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        r = json.loads(raw)
        weighted = round(sum(r[k] * w for k, w in WEIGHTS.items()), 2)   # per-topic weighted (AC-03)
        r["topic_score"] = weighted
        r["topic"] = topic
        r["answered"] = bool(r.get("answered", True))
        flag = "" if r["answered"] else "  ⚠️ UNANSWERED"
        print(f"   📐 {topic}: T{r['technical_accuracy']} D{r['depth']} C{r['clarity']} "
              f"P{r['problem_solving']} → {weighted}{flag}")
        return r
    except Exception as e:
        print(f"❌ _score_topic({topic}): {e}")
        return {"topic": topic, "technical_accuracy": 5, "depth": 5, "clarity": 5,
                "problem_solving": 5, "topic_score": 5.0, "answered": True,
                "note": "scoring failed — defaulted"}


def _recommendation(composite: float) -> str:
    """Map composite to the four required bands (US-AG-08 AC-03)."""
    if composite >= 8.0:   return "Strongly Recommended"
    if composite >= 6.5:   return "Recommended"
    if composite >= 5.0:   return "Needs Further Review"
    return "Not Recommended"


def _summarize(topic_scores: list, composite: float, recommendation: str,
               candidate: dict, status: str) -> dict:
    """One more LLM call to write strengths, gaps, and a justification citing the scores."""
    breakdown = "\n".join(f"- {t['topic']}: {t['topic_score']}/10 ({t.get('note','')})"
                          for t in topic_scores)
    incomplete = status != "completed"
    prompt = f"""You are writing the narrative section of a hiring evaluation report. Be specific and
reference the per-topic results. Do not invent facts not in the breakdown.

CANDIDATE: {candidate.get('name') or 'Unknown'}  ROLE: {candidate.get('role') or 'Unspecified'}
OVERALL COMPOSITE: {composite}/10
RECOMMENDATION: {recommendation}
SESSION STATUS: {status}{"  (INTERVIEW INCOMPLETE — note this)" if incomplete else ""}

PER-TOPIC BREAKDOWN:
{breakdown}

Reply ONLY valid JSON, no markdown:
{{"strengths":"<2-3 sentences on what was strong, citing topics>",
"gaps":"<2-3 sentences on weaknesses/gaps, citing topics>",
"justification":"<2-3 sentences explaining the recommendation, referencing specific scores>"}}"""
    try:
        resp = groq_client.chat.completions.create(
            model=EVAL_MODEL, messages=[{"role": "user", "content": prompt}],
            max_tokens=400, temperature=0.3)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"❌ _summarize(): {e}")
        return {"strengths": "See per-topic scores.", "gaps": "See per-topic scores.",
                "justification": f"Composite {composite}/10 → {recommendation}."}


def evaluate_session(session_id: str, status: str = "completed") -> dict:
    """
    MAIN ENTRY. Reads the transcript, scores per topic, computes composite + recommendation,
    writes the report row, and returns the report dict. Called once at interview end.
    """
    print(f"\n🧮 Evaluating session {session_id} (status={status})...")
    answers = db.read_answers(session_id)
    if not answers:
        print("⚠️ no answers to evaluate")
        return {}

    candidate = db.get_candidate_for_session(session_id)
    topics = _group_by_topic(answers)
    # only score topics that actually got an answer row
    all_scores = [_score_topic(t, b) for t, b in topics.items() if b["answer"]]
    if not all_scores:
        print("⚠️ no answered topics")
        return {}

    # Split: genuinely-answered vs near-empty (the scorer's "answered" flag)
    answered  = [t for t in all_scores if t.get("answered", True)]
    unanswered = [t for t in all_scores if not t.get("answered", True)]
    unanswered_topics = [t["topic"] for t in unanswered]

    scored = answered or all_scores   # safety: if somehow all unanswered, fall back to all
    # Composite = average of ANSWERED topics only (AC-04, adjusted per decision)
    composite = round(sum(t["topic_score"] for t in scored) / len(scored), 2)
    recommendation = _recommendation(composite)

    # Cap: 3+ unanswered topics → force "Needs Further Review" regardless of composite
    capped = False
    if len(unanswered) >= 3 and recommendation in ("Strongly Recommended", "Recommended"):
        recommendation = "Needs Further Review"
        capped = True
        print(f"🚧 {len(unanswered)} unanswered topics → recommendation capped to Needs Further Review")

    narrative = _summarize(scored, composite, recommendation, candidate, status)

    # Dimension averages across ANSWERED topics (headline metrics)
    def davg(dim): return round(sum(t[dim] for t in scored) / len(scored), 2)

    gaps_text = narrative["gaps"]
    if unanswered_topics:
        gaps_text += (f" Topics with no substantive answer (excluded from the score): "
                      f"{', '.join(unanswered_topics)}.")

    rec_final = recommendation if status == "completed" else f"{recommendation} (Incomplete Session)"

    report = {
        "technical_accuracy": davg("technical_accuracy"),
        "depth": davg("depth"),
        "clarity": davg("clarity"),
        "problem_solving": davg("problem_solving"),
        "overall_score": composite,
        "recommendation": rec_final,
        "strengths": narrative["strengths"],
        "gaps": gaps_text,
        "justification": narrative["justification"],
    }
    db.save_report(session_id, report)

    full = {**report, "session_id": session_id, "candidate": candidate, "status": status,
            "scored_topics": len(scored), "unanswered_topics": unanswered_topics,
            "recommendation_capped": capped,
            "per_topic": all_scores, "generated_at": datetime.now().isoformat()}
    fn = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, "w") as f:
        json.dump(full, f, indent=2)
    print(f"📄 Report: composite {composite}/10 (from {len(scored)} answered topics) → "
          f"{rec_final}  (saved {fn})")
    return full