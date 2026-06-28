# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A two-layer data feedback system for construction vehicle detection: a TypeScript PI Agent orchestrates two Python FastAPI services via HTTP. The agent provides an interactive REPL where users can verify YOLO detections with LLM analysis and run multi-dimensional dataset quality analysis.

## Architecture

```
User → PI Agent (TypeScript, agent.ts)
         ├── tools.ts (9 HTTP tool wrappers)
         │    ├── 4 detection tools → detection-service (:8001)
         │    └── 5 analysis tools  → data-analysis-service (:8002)
         └── .pi/skills/ (3 SKILL.md files define workflows)

detection-service (:8001)
  ├─ /api/verify        — Remote YOLO detection + LLM verification
  ├─ /api/correct       — Detection + LLM correction + comparison images
  ├─ /api/verify_direct — LLM verification only (skip YOLO)
  └─ /api/verify_batch  — Batch verification
  External deps: Remote YOLO API (192.168.99.180:49080), MiMo LLM

data-analysis-service (:8002)
  ├─ /api/profile       — Dataset profiling (class distribution, bbox stats)
  ├─ /api/analyze/single — Single image 5-dimension analysis
  ├─ /api/analyze/batch  — Batch analysis with LLM attribution
  └─ /api/export        — Export results
  External deps: BGE-VL-large (local GPU), MiMo LLM
```

**Key design decisions:**
- Agent LLM (GPT-5.5) handles reasoning/tool selection; service LLM (MiMo v2.5) handles verification/correction prompts
- detection-service uses function calling (`report_verification`, `report_missed_bbox` tools) for structured LLM output
- data-analysis-service precomputes training set distributions at startup, then compares per-image against them
- Both services read shared `config.yaml` from project root (walks up directory tree)

## Commands

### Start Python Services (conda `zzq` env required)

```bash
# Detection service (port 8001)
cd services/detection-service && bash start.sh
# Or directly: python app.py

# Data analysis service (port 8002)
cd services/data-analysis-service && uvicorn app:app --host 0.0.0.0 --port 8002
```

### Run PI Agent (Node 22 via nvm)

```bash
npm run agent          # node --import tsx/esm agent.ts
npm run agent:pi       # pi run agent.ts
```

### Tests

```bash
# Full pipeline test (detect → verify → correct → attribution)
python tests/test_services.py

# Detection service tests
cd services/detection-service && python test/test_service.py
cd services/detection-service && python -m pytest test/

# Secondary review evaluation
cd services/detection-service && python test/evaluate_secondary_review.py

# Analysis service health
curl http://localhost:8002/health

# Agent integration test
node tests/test_agent.ts
```

## Configuration

All config lives in `config.yaml` (project root). Key sections:
- `server.ports` — detection=8001, data-analysis=8002
- `llm` — MiMo v2.5 via OpenAI-compatible API (protocol, api_url, api_key, model)
- `detection.api_url` — Remote YOLO endpoint
- `semantic.model_path` — BGE-VL-large local path (GPU required)
- `classes` — 9 vehicle classes (wajueji, chanche, dazhuangji, yaluji, diaoche, gaokongche, youguanche, yunshuche, other)

PI Agent config in `.pi/`:
- `settings.json` — provider/model selection (openai, gpt-5.5)
- `auth.json` — API keys
- `models.json` — custom model definitions
- `skills/` — 3 workflow skills (detection-review, detection-correction, data-analysis)

## Code Structure Notes

- `services/detection-service/llm/` — LLM client abstraction with factory pattern: `create_llm_client("openai"|"anthropic"|"ollama")`
- `services/detection-service/scenarios/` — Vehicle class definitions and scenario-specific configuration
- `services/detection-service/prompts/` — All LLM prompts are in Chinese; verification prompt includes vehicle-specific visual features
- `services/detection-service/tools/function_calling.py` — JSON schemas for function calling (used by verifier.py and correction_service.py)
- `services/data-analysis-service/analyzers/` — 6 semantic analyzers extend `BaseSemanticAnalyzer` (BGE-VL-large embeddings + cosine similarity); `ClassAnalyzer` is standalone; `LLMAttributionAnalyzer` aggregates all dimensions
- `services/data-analysis-service/pipeline.py` — `AnalysisPipeline` orchestrates: precompute training distributions → per-image analysis → LLM attribution

## Environment

- Python: conda `zzq` environment
- Node.js: >= 22.19.0 (via nvm)
- CUDA GPU required for BGE-VL-large embeddings in data-analysis-service
- Windows paths (Git Bash for shell scripts)
