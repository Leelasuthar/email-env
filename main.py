from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
import random
import uvicorn

app = FastAPI(title="Email Triage Environment", version="1.0.0")

# ─────────────────────────────────────────────
# Typed Models
# ─────────────────────────────────────────────

class EmailAction(BaseModel):
    message: str

class EmailObservation(BaseModel):
    email: str
    task: str
    echoed_message: str
    is_critical: bool = False

class StepResult(BaseModel):
    observation: EmailObservation
    reward: float
    done: bool
    info: Optional[dict] = {}

class EnvState(BaseModel):
    step_count: int
    task: str
    email: str
    is_critical: bool
    history: List[str]

# ─────────────────────────────────────────────
# Email Dataset — 12 normal + 4 critical
# ─────────────────────────────────────────────

NORMAL_EMAILS = [
    {
        "text": "Hi, could you please send me the invoice for order #4521? We need it for our records.",
        "category": "invoice", "department": "finance",
        "reply_keywords": ["invoice", "attached", "send"],
        "reply_negative": ["sorry", "apologize", "complaint"],
        "workflow_keywords": ["send invoice", "finance team", "billing"],
        "is_critical": False,
    },
    {
        "text": "Our accounts team is requesting invoice #7823 urgently. The payment deadline is tomorrow.",
        "category": "invoice", "department": "finance",
        "reply_keywords": ["invoice", "urgent", "send"],
        "reply_negative": ["sorry", "apologize"],
        "workflow_keywords": ["send invoice", "finance", "accounts"],
        "is_critical": False,
    },
    {
        "text": "Please resend invoice for the March services. We seem to have misplaced the original.",
        "category": "invoice", "department": "finance",
        "reply_keywords": ["invoice", "resend", "attached"],
        "reply_negative": ["sorry", "complaint"],
        "workflow_keywords": ["send invoice", "finance", "resend"],
        "is_critical": False,
    },
    {
        "text": "I am very disappointed. My order was supposed to arrive last week and I still have not received it.",
        "category": "complaint", "department": "support",
        "reply_keywords": ["sorry", "apologize", "resolve"],
        "reply_negative": ["invoice", "finance", "payment"],
        "workflow_keywords": ["escalate", "resolve", "support team"],
        "is_critical": False,
    },
    {
        "text": "This is unacceptable! I received the wrong product and nobody from your team is responding.",
        "category": "complaint", "department": "support",
        "reply_keywords": ["apologize", "sorry", "fix", "resolve"],
        "reply_negative": ["invoice", "finance"],
        "workflow_keywords": ["escalate", "resolve", "urgent"],
        "is_critical": False,
    },
    {
        "text": "Your customer service is terrible. I have been waiting 2 weeks for a refund and nothing has happened.",
        "category": "complaint", "department": "support",
        "reply_keywords": ["sorry", "apologize", "refund", "resolve"],
        "reply_negative": ["invoice", "payment status"],
        "workflow_keywords": ["escalate", "refund", "resolve"],
        "is_critical": False,
    },
    {
        "text": "I would like to request 5 days of annual leave from 15th to 19th April. Please approve.",
        "category": "leave", "department": "hr",
        "reply_keywords": ["approved", "leave", "noted"],
        "reply_negative": ["sorry", "invoice", "payment"],
        "workflow_keywords": ["approve", "update system", "hr"],
        "is_critical": False,
    },
    {
        "text": "I need to take medical leave for 3 days starting Monday due to a health issue.",
        "category": "leave", "department": "hr",
        "reply_keywords": ["approved", "medical", "noted", "get well"],
        "reply_negative": ["invoice", "complaint"],
        "workflow_keywords": ["approve", "update system", "medical"],
        "is_critical": False,
    },
    {
        "text": "Requesting maternity leave starting next month for 3 months as per company policy.",
        "category": "leave", "department": "hr",
        "reply_keywords": ["approved", "maternity", "congratulations", "noted"],
        "reply_negative": ["invoice", "payment"],
        "workflow_keywords": ["approve", "update system", "hr policy"],
        "is_critical": False,
    },
    {
        "text": "We sent our invoice 30 days ago and payment is still pending. Could you check the status?",
        "category": "payment", "department": "finance",
        "reply_keywords": ["processing", "payment", "status", "check"],
        "reply_negative": ["sorry", "complaint", "leave"],
        "workflow_keywords": ["process payment", "finance", "accounts"],
        "is_critical": False,
    },
    {
        "text": "This is a reminder that payment for invoice #2233 is overdue by 15 days. Please advise.",
        "category": "payment", "department": "finance",
        "reply_keywords": ["payment", "processing", "overdue", "resolve"],
        "reply_negative": ["sorry", "complaint"],
        "workflow_keywords": ["process payment", "notify", "finance"],
        "is_critical": False,
    },
    {
        "text": "Can you confirm if the bank transfer for our services last month has been processed?",
        "category": "payment", "department": "finance",
        "reply_keywords": ["confirmed", "payment", "transfer", "processed"],
        "reply_negative": ["sorry", "complaint", "leave"],
        "workflow_keywords": ["confirm payment", "finance", "transfer"],
        "is_critical": False,
    },
]

CRITICAL_EMAILS = [
    {
        "text": "We are furious with your service. If this is not resolved by tomorrow we will take legal action and contact our lawyers immediately.",
        "category": "complaint", "department": "support",
        "reply_keywords": ["sorry", "apologize", "resolve", "urgent"],
        "reply_negative": ["invoice", "finance"],
        "workflow_keywords": ["escalate", "legal", "management", "human review"],
        "is_critical": True,
        "critical_reason": "legal threat detected",
    },
    {
        "text": "Your company owes us payment for invoice #9981 which is now overdue by 90 days. We demand immediate payment or we will proceed with legal proceedings.",
        "category": "payment", "department": "finance",
        "reply_keywords": ["payment", "resolve", "urgent", "processing"],
        "reply_negative": ["sorry", "leave", "complaint"],
        "workflow_keywords": ["process payment", "legal", "management", "human review"],
        "is_critical": True,
        "critical_reason": "legal threat + major overdue payment",
    },
    {
        "text": "I am escalating this to your CEO and contacting consumer protection authorities. Your team has failed to respond for 30 days and I want a full refund NOW.",
        "category": "complaint", "department": "support",
        "reply_keywords": ["apologize", "sorry", "resolve", "refund"],
        "reply_negative": ["invoice", "finance"],
        "workflow_keywords": ["escalate", "management", "ceo", "human review"],
        "is_critical": True,
        "critical_reason": "CEO escalation + regulatory threat",
    },
    {
        "text": "This is our final notice before we engage our legal team. Payment of $50,000 for services rendered in Q1 remains outstanding for 120 days.",
        "category": "payment", "department": "finance",
        "reply_keywords": ["payment", "urgent", "resolve", "processing"],
        "reply_negative": ["sorry", "leave"],
        "workflow_keywords": ["process payment", "legal", "human review", "management"],
        "is_critical": True,
        "critical_reason": "final legal notice + large amount",
    },
]

ALL_EMAILS = NORMAL_EMAILS + CRITICAL_EMAILS

# ─────────────────────────────────────────────
# Critical email detector
# ─────────────────────────────────────────────

CRITICAL_SIGNALS = [
    "legal action", "lawyer", "lawyers", "legal team", "legal proceedings",
    "sue", "lawsuit", "authorities", "ceo", "consumer protection",
    "final notice", "90 days", "120 days", "furious", "demand immediate",
]

def is_critical_email(text: str) -> bool:
    t = text.lower()
    return any(signal in t for signal in CRITICAL_SIGNALS)

# ─────────────────────────────────────────────
# Reward Functions
# FIX: floors changed from 0.1 → 0.0 so bad actions are genuinely penalised
# ─────────────────────────────────────────────

def grade_classification(message: str, email: dict) -> float:
    msg = message.lower()

    # FIX: penalise empty or nonsensical (< 2 chars) actions
    if len(msg.strip()) < 2:
        return 0.0

    category   = email["category"]
    department = email["department"]

    if department in msg:                        return 1.0
    if category in msg and len(msg.split()) >= 2: return 0.8
    if category in msg:                          return 0.6

    related = {
        "invoice":   ["billing", "accounts", "accounting"],
        "complaint": ["customer", "service", "issue", "problem"],
        "leave":     ["human resources", "absence", "time off"],
        "payment":   ["billing", "accounts", "accounting", "vendor"],
    }
    for word in related.get(category, []):
        if word in msg:
            return 0.5

    return 0.0   # FIX: was 0.1 — completely wrong answer now gets 0


def grade_reply(message: str, email: dict) -> float:
    msg = message.lower()

    # FIX: penalise empty or nonsensical actions
    if len(msg.strip()) < 2:
        return 0.0

    positive = email["reply_keywords"]
    negative = email["reply_negative"]

    penalty = sum(1 for w in negative if w in msg)
    if penalty >= 2:
        return 0.0   # FIX: was 0.1

    hits = sum(1 for w in positive if w in msg)
    if hits == 0:   return max(0.0, 0.2 - 0.1 * penalty)   # FIX: floor 0.0
    elif hits == 1: return max(0.0, 0.5 - 0.1 * penalty)
    elif hits == 2: return max(0.0, 0.75 - 0.1 * penalty)
    else:           return max(0.0, min(1.0, 0.9 + 0.05 * (hits - 2)) - 0.1 * penalty)


def grade_workflow(message: str, email: dict) -> float:
    msg = message.lower()

    # FIX: penalise empty or nonsensical actions
    if len(msg.strip()) < 2:
        return 0.0

    hits = sum(1 for w in email["workflow_keywords"] if w in msg)
    if hits == 0:   return 0.0   # FIX: was 0.2
    elif hits == 1: return 0.55
    elif hits == 2: return 0.8
    else:           return 1.0


def grade_human_review(message: str, email: dict) -> float:
    msg = message.lower()

    # FIX: penalise empty or nonsensical actions
    if len(msg.strip()) < 2:
        return 0.0

    human_signals  = ["human review", "human", "escalate to manager", "manager",
                      "management", "not for ai", "requires human", "flag"]
    legal_signals  = ["legal", "lawyer", "authorities", "ceo"]
    urgent_signals = ["urgent", "critical", "immediate", "priority"]

    human_hits  = sum(1 for w in human_signals  if w in msg)
    legal_hits  = sum(1 for w in legal_signals  if w in msg)
    urgent_hits = sum(1 for w in urgent_signals if w in msg)

    total = human_hits + legal_hits + urgent_hits

    # FIX: agent handles critical email alone → real penalty (was 0.1)
    if total == 0:
        return 0.0

    if human_hits >= 1 and (legal_hits >= 1 or urgent_hits >= 1): return 1.0
    elif human_hits >= 1:  return 0.8
    elif total >= 2:       return 0.7
    elif total == 1:       return 0.4
    return 0.0   # FIX: was 0.1


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

# FIX: max steps enforced inside env to prevent infinite loops
MAX_ENV_STEPS = 10

class EmailEnv:
    def __init__(self):
        self.step_count    = 0
        self.task          = "classification"
        self.current_email: dict = {}
        self.history: List[str] = []

    def reset(self, task: str = None) -> StepResult:
        self.step_count = 0
        self.history    = []

        if task == "human_review":
            self.current_email = random.choice(CRITICAL_EMAILS)
            self.task = "human_review"
        elif task in ("classification", "reply", "workflow"):
            self.current_email = random.choice(NORMAL_EMAILS)
            self.task = task
        else:
            # default: random behaviour
            self.current_email = random.choice(ALL_EMAILS)
            self.task = "human_review" if self.current_email["is_critical"] else "classification"

        return StepResult(
            observation=EmailObservation(
                email=self.current_email["text"],
                task=self.task,
                echoed_message="",
                is_critical=self.current_email["is_critical"],
            ),
            reward=0.0,
            done=False,
            info={"critical_reason": self.current_email.get("critical_reason", "")}
                 if self.current_email["is_critical"] else {},
        )

    def step(self, action: EmailAction) -> StepResult:
        self.step_count += 1
        message = action.message.lower().strip()

        # FIX: penalise loop/repetition — same action sent twice in a row
        loop_penalty = 0.0
        if self.history and self.history[-1] == message:
            loop_penalty = 0.2

        # FIX: hard stop if agent exceeds MAX_ENV_STEPS (prevents infinite loops)
        if self.step_count > MAX_ENV_STEPS:
            self.history.append(message)
            return StepResult(
                observation=EmailObservation(
                    email=self.current_email["text"],
                    task=self.task,
                    echoed_message=message,
                    is_critical=self.current_email["is_critical"],
                ),
                reward=0.0,
                done=True,
                info={"warning": "max steps exceeded"},
            )

        self.history.append(message)

        reward = 0.0
        done   = False

        if self.task == "human_review":
            reward = grade_human_review(message, self.current_email)
            self.task = "reply"

        elif self.task == "reply":
            reward = grade_reply(message, self.current_email)
            self.task = "workflow"

        elif self.task == "workflow":
            reward = grade_workflow(message, self.current_email)
            done = True

        elif self.task == "classification":
            reward = grade_classification(message, self.current_email)
            self.task = "reply"

        # FIX: apply loop penalty
        reward = max(0.0, round(reward - loop_penalty, 4))

        return StepResult(
            observation=EmailObservation(
                email=self.current_email["text"],
                task=self.task,
                echoed_message=message,
                is_critical=self.current_email["is_critical"],
            ),
            reward=reward,
            done=done,
        )

    def state(self) -> EnvState:
        return EnvState(
            step_count=self.step_count,
            task=self.task,
            email=self.current_email.get("text", ""),
            is_critical=self.current_email.get("is_critical", False),
            history=self.history,
        )


# ─────────────────────────────────────────────
# Shared instance
# ─────────────────────────────────────────────
env = EmailEnv()

# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "email-triage"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "email-env",
        "description": (
            "An OpenEnv RL environment simulating enterprise email triage "
            "with human-in-the-loop detection for critical emails."
        ),
        "version": "1.0.0",
    }

@app.get("/schema")
def get_schema():
    return {
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "capabilities": {"reset": True, "step": True, "state": True},
        },
    }

@app.post("/reset", response_model=StepResult)
def reset(task: Optional[str] = None):
    return env.reset(task=task)

@app.post("/step", response_model=StepResult)
def step(action: EmailAction):
    return env.step(action)

@app.get("/state", response_model=EnvState)
def state():
    return env.state()

def serve():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    serve()