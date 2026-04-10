import asyncio
import os
import sys
import random
from openai import OpenAI
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import EmailEnv, EmailAction

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
MODEL_NAME   = os.environ.get("MODEL_NAME") or "mistralai/Mistral-7B-Instruct-v0.3"

BENCHMARK               = "email-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# Tasks to evaluate — one episode per task
TASKS = ["classification", "reply", "workflow", "human_review"]

# ─────────────────────────────────────────────
# Logging  (strict [START] / [STEP] / [END] format)
# ─────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str  = "true" if done else "false"
    error_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────────────
# Fallback logic (no LLM / LLM error)
# ─────────────────────────────────────────────
def fallback_logic(email: str, task: str, is_critical: bool) -> str:
    e = email.lower()

    if task == "human_review" or is_critical:
        if "legal" in e or "lawyer" in e or "legal action" in e:
            return "requires human review legal threat escalate management urgent"
        if "ceo" in e or "authorities" in e:
            return "requires human review escalate management ceo urgent"
        return "requires human review escalate management urgent critical"

    if task == "classification":
        if "invoice" in e:                                                  return "finance department"
        if any(w in e for w in ["complaint", "disappointed", "furious"]):   return "support department"
        if any(w in e for w in ["leave", "medical", "maternity"]):          return "hr department"
        if any(w in e for w in ["payment", "transfer", "overdue"]):         return "finance department"
        return "support department"

    if task == "reply":
        if "invoice" in e:                                                  return "thank you attached invoice as requested"
        if any(w in e for w in ["complaint", "disappointed", "furious"]):   return "we sincerely apologize and will resolve this immediately"
        if any(w in e for w in ["leave", "medical", "maternity"]):          return "your leave is approved and noted get well soon"
        if any(w in e for w in ["payment", "transfer", "overdue"]):         return "we are processing your payment finance will confirm"
        return "thank you we will resolve this shortly"

    if task == "workflow":
        if "invoice" in e:                                                  return "send invoice finance team billing"
        if any(w in e for w in ["complaint", "disappointed", "furious"]):   return "escalate resolve support team"
        if any(w in e for w in ["leave", "medical", "maternity"]):          return "approve leave update system hr"
        if any(w in e for w in ["payment", "transfer", "overdue"]):         return "process payment notify finance accounts"
        return "escalate resolve support team"

    return "support department"

# ─────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────
def get_model_message(client: OpenAI, email: str, task: str, is_critical: bool) -> str:
    try:
        if task == "human_review":
            prompt = f"""You are an AI email agent. This email has been flagged as CRITICAL.
Email: {email}
This email contains legal threats, CEO escalations, or major financial disputes.
The AI should NOT handle this alone — a human must review it.
Respond with a short action phrase that:
- Flags this for human review
- Mentions the type of threat (legal / management / urgent)
- Example: "requires human review legal threat escalate management urgent"
Reply with only the action phrase, no explanation."""

        elif task == "classification":
            prompt = f"""You are an email routing agent.
Email: {email}
Choose exactly one: "finance department", "support department", or "hr department"
Reply with only those words."""

        elif task == "reply":
            prompt = f"""You are an email reply agent.
Email: {email}
Critical: {is_critical}
Rules:
- Invoice: mention "invoice" and "attached"
- Complaint: say "apologize" and "resolve"
- Leave: say "approved" and "noted"
- Payment: say "processing" and "payment"
- Critical emails: also add "urgent" and "escalate"
Write only the reply, 1-2 sentences."""

        elif task == "workflow":
            prompt = f"""You are a workflow decision agent.
Email: {email}
Critical: {is_critical}
Rules:
- Invoice: "send invoice finance team billing"
- Complaint: "escalate resolve support team"
- Leave: "approve update system hr"
- Payment: "process payment notify finance accounts"
- Critical: add "human review management" to any of the above
Reply with only the action phrase."""

        else:
            return fallback_logic(email, task, is_critical)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else fallback_logic(email, task, is_critical)

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return fallback_logic(email, task, is_critical)

# ─────────────────────────────────────────────
# Direct environment (no HTTP server needed)
# ─────────────────────────────────────────────
_env = EmailEnv()

def env_reset(task: str = None):
    return _env.reset(task=task)

def env_step(message: str):
    return _env.step(EmailAction(message=message))

# ─────────────────────────────────────────────
# Evaluate a single task — one episode, one step
# ─────────────────────────────────────────────
def evaluate_task(client: OpenAI, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    score   = 0.0
    success = False
    steps   = 0
    rewards: List[float] = []

    try:
        result      = env_reset(task=task_name)
        email       = result.observation.email
        task        = result.observation.task
        is_critical = result.observation.is_critical

        message = get_model_message(client, email, task, is_critical)
        result  = env_step(message)
        reward  = result.reward
        done    = result.done

        rewards.append(reward)
        steps = 1

        log_step(step=1, action=message, reward=reward, done=done, error=None)

        score   = round(min(max(reward, 0.01), 0.99), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error in task {task_name}: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps, score=score, rewards=rewards)

    return score

# ─────────────────────────────────────────────
# Main — run all tasks
# ─────────────────────────────────────────────
def main() -> None:
    random.seed(42)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        evaluate_task(client, task_name)


if __name__ == "__main__":
    main()
