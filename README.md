---
title: Email Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
tags: [openenv]
pinned: false
---
# Email Triage Environment

An OpenEnv RL environment simulating enterprise email triage with
**human-in-the-loop** detection for critical emails.

## Baseline scores

| Agent | Score |
|-------|-------|
| Rule-based fallback | 0.74 |
| LLM agent (Mistral-7B) | 0.83 |

## Environment description

The agent receives a business email and must complete tasks based on whether
the email is normal or critical.

### Normal email flow (3 tasks)

| Task | Difficulty | Description |
|------|-----------|-------------|
| classification | Easy | Route to finance / support / hr |
| reply | Medium | Write contextually correct reply |
| workflow | Hard | Decide next business action |

### Critical email flow (3 tasks)

| Task | Difficulty | Description |
|------|-----------|-------------|
| human_review | Hard | Detect critical email, flag for human |
| reply | Medium | Acknowledge urgency appropriately |
| workflow | Hard | Escalate to management + legal |

## Email scenarios (16 total)

| Category | Count | Type |
|----------|-------|------|
| Invoice | 3 | Normal |
| Complaint | 3 | Normal |
| HR / Leave | 3 | Normal |
| Payment | 3 | Normal |
| Legal threats | 2 | Critical |
| CEO escalations | 1 | Critical |
| Major overdue payment | 1 | Critical |

## Human-in-the-loop design

Critical emails are detected by signals including:

- Legal language: `"legal action"`, `"lawyer"`, `"lawsuit"`
- Executive escalation: `"ceo"`, `"authorities"`
- Severe overdue: 90+ days outstanding
- Regulatory threats: `"consumer protection"`

When a critical email is detected, the agent must respond with escalation
language (human review, management, urgent) rather than handling it alone.
Attempting to handle a critical email without escalation is penalised with a
reward of 0.0.

## Reward design

### Classification (easy)
- 1.0 → correct department named
- 0.6–0.8 → category keyword present
- 0.5 → related synonym matched
- 0.0 → completely wrong or empty

### Reply (medium)
- Rewards context-correct keywords, penalises wrong-context words
- 0.0 for empty actions or 2+ negative keyword matches

### Workflow (hard)
- Rewards multi-keyword correct action phrases
- 0.0 for no keyword match or empty action

### Human review (hard)
- 1.0 → escalation + legal/urgent signal both present
- 0.0 → agent tries to handle critical email alone or sends empty action
- Repeated identical actions incur a 0.2 loop penalty

## Action space

- Type: text (string)
- Normal examples: `"finance department"`, `"we apologize and will resolve this"`
- Critical examples: `"requires human review legal threat escalate management urgent"`

## Observation space

| Field | Type | Description |
|-------|------|-------------|
| email | string | Full incoming email text |
| task | string | classification / reply / workflow / human_review |
| echoed_message | string | Agent's last action |
| is_critical | boolean | True if email requires human intervention |

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860

export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=your_token_here
python inference.py
```

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Send action, get reward |
| `/state` | GET | Current env state |

## Environment variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM API base URL |
| MODEL_NAME | Model identifier |
| HF_TOKEN | Hugging Face API token (free) |