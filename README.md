# VoiceAiBot

A Darwin Gödel Machine-inspired platform that evolves a debt collection voice agent. The system continuously improves agent performance by simulating conversations with diverse borrower personas, scoring results across multiple metrics, and rewriting its own prompts — without human intervention.

---

## Overview

The project is split into two parts:

**Part 1 — Base Voice Agent**
- Real-time voice bot built with [Pipecat](https://github.com/pipecat-ai/pipecat): Deepgram STT → Gemini LLM → Cartesia TTS
- Text-based simulation against 5 LLM-simulated borrower personas
- Automated LLM evaluation across 3 weighted metrics

**Part 2 — Self-Evolving Agent**
- LLM-guided mutation of agent prompts based on failure analysis
- Persistent archive tracking every version with full provenance
- Autonomous evolution loop with configurable termination conditions

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Voice Pipeline                    │
│  WebSocket → Deepgram STT → Gemini LLM → Cartesia   │
│              TTS → HangupDetector → WebSocket       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                    Evolution Loop                   │
│                                                     │
│  Best Agent → Weak Session Analysis                 │
│      ↓                                              │
│  Gemini proposes prompt mutation                    │
│      ↓                                              │
│  Simulate (N runs × 5 personas)                     │
│      ↓                                              │
│  Evaluate (goal + quality + compliance)             │
│      ↓                                              │
│  fitness > parent → Promote │ else → Fail           │
│      ↓                                              │
│  Archive (version, score, rationale, lineage)       │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
VoiceAiBot/
├── configs/
│   ├── agents/           # Agent YAML configs (v1, v2, ...) + archive.json
│   ├── personas/         # 5 borrower persona YAMLs
│   └── evaluation/       # Scoring rubric
├── src/
│   ├── agent/            # Config models, prompt builder
│   ├── voice/            # Pipecat pipeline, WebSocket server, serializer
│   ├── simulation/       # Conversation runner, session recorder
│   ├── evaluation/       # LLM scorer, metrics, report generator
│   ├── evolution/        # Archive, mutation engine, failure analyzer
│   └── utils/            # Gemini client, logger
├── scripts/
│   ├── run_voice_server.py
│   ├── run_simulation.py
│   ├── run_evaluation.py
│   └── run_evolution.py
└── results/              # Saved session JSONs and evaluation reports
```

---

## Setup

**Requirements:** Python 3.11+, Google Cloud project with Vertex AI enabled

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-east5
DEEPGRAM_API_KEY=your-deepgram-key
CARTESIA_API_KEY=your-cartesia-key
```

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

---

## Usage

### Run a voice conversation

Start the WebSocket server for an agent version:

```bash
python scripts/run_voice_server.py --agent v1
```

Connect a WebSocket client to `ws://localhost:8765`. The agent will greet the borrower and handle the conversation end-to-end with voice input/output.

Options:
```
--agent   Agent version to load (default: v1)
--host    Server host (default: localhost)
--port    Server port (default: 8765)
```

---

### Run text simulations

Simulate conversations between the agent and LLM-backed personas:

```bash
python scripts/run_simulation.py --agent v1 --runs 15
```

Run against a specific persona:

```bash
python scripts/run_simulation.py --agent v1 --runs 5 --persona angry_defaulter
```

Sessions are saved to `results/v1/*.json`.

Options:
```
--agent    Agent version (default: v1)
--runs     Number of simulations (default: 5)
--persona  Specific persona ID (default: cycle through all 5)
```

---

### Evaluate simulations

Score saved sessions and generate a report:

```bash
python scripts/run_evaluation.py --agent v1
```

Prints per-session scores, per-persona summary, and overall fitness score. Writes `results/v1/evaluation_report.json` and updates `fitness_score` in the agent YAML.

Options:
```
--agent   Agent version to evaluate (default: v1)
```

---

### Run the evolution loop

Autonomously evolve the agent over multiple generations:

```bash
python scripts/run_evolution.py
```

Each generation:
1. Selects the best promoted agent
2. Extracts its weakest sessions (bottom 25%)
3. Asks Gemini to propose targeted prompt mutations
4. Runs simulations on the mutated agent
5. Promotes if fitness improves; marks failed otherwise
6. Stops when fitness reaches the threshold or progress plateaus

Options:
```
--start-version          Seed agent version (default: v1)
--max-generations        Maximum generations to run (default: 10)
--success-threshold      Fitness score to stop at (default: 4.9)
--plateau-patience       Stop after N non-improving generations (default: 3)
--sim-runs               Simulations per candidate (default: 15)
--allow-llm-param-mutation  Also evolve temperature/max_tokens
--dry-run                Propose mutation only, skip simulation
```

---

## Borrower Personas

| ID | Name | Agreement Probability | Trait |
|----|------|-----------------------|-------|
| `angry_defaulter` | Marcus | 25% | Hostile and defensive |
| `evasive_defaulter` | Sarah | 30% | Makes excuses, deflects |
| `curious_defaulter` | David | 55% | Asks many questions |
| `cooperative_defaulter` | Lisa | 70% | Friendly and apologetic |
| `distressed_defaulter` | James | 40% | Emotionally overwhelmed |

---

## Evaluation Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| `goal_completion` | 40% | Did the agent secure a repayment commitment? |
| `conversational_quality` | 35% | Natural, empathetic, effective communication? |
| `compliance` | 25% | Avoided forbidden phrases and threats? |

Scores are 1–5 per metric. **Fitness score** = mean weighted total across all sessions. Auto-caps compliance at 2.0 if forbidden phrases are detected via regex.

---

## Agent Configuration

Each agent version is a YAML file at `configs/agents/agent_v{N}.yaml`:

```yaml
version: v1
description: Baseline debt collection agent
fitness_score: 4.6889
generation: 0
parent_version: null

llm:
  provider: vertex-gemini
  model: gemini-2.5-pro
  temperature: 0.3
  max_tokens: 2000

tts:
  provider: cartesia
  voice_id: e07c00bc-4134-4eae-9ea4-1a55fb45746b

stt:
  provider: deepgram
  model: nova-2
  language: en-US

prompt:
  persona_header: "You are Alex, a professional loan recovery specialist..."
  goal_statement: "Arrange a repayment commitment..."
  behavioral_guidelines: "Remain calm and empathetic..."
  compliance_rules: "NEVER say: 'sue you', 'jail', 'arrest'..."
  conversation_style: "2-3 sentences per turn..."
  opening_script: "Hello, may I speak with [BORROWER_NAME]?..."

hangup_phrases:
  - goodbye
  - take care
  - have a good day
```

The `prompt` section has 6 independently mutable subsections. The evolution loop rewrites 1–3 of them per generation while keeping `compliance_rules` locked.

---

## Evolution Archive

All agent versions are tracked in `configs/agents/archive.json` with:
- `fitness_score` — mean weighted evaluation score
- `status` — `promoted`, `failed`, or `pending`
- `parent_version` — which agent it was evolved from
- `mutation_rationale` — why this mutation was proposed
- `failure_addressed` — what failure pattern it targeted
- `mutations_applied` — which prompt sections were changed
- `per_metric_scores` / `per_persona_scores` — breakdown

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Voice pipeline | [Pipecat](https://github.com/pipecat-ai/pipecat) |
| STT | Deepgram Nova-2 |
| LLM (agent + evolution) | Google Vertex AI Gemini 2.5 Pro |
| LLM (personas) | Google Vertex AI Gemini 2.5 Flash |
| TTS | Cartesia |
| VAD | Silero |
| Config | Pydantic + YAML |
| CLI | Click + Rich |
