# FinePrint
Fine-tuned Qwen3.5-4B (QLoRA SFT -> GRPO) turns advisor notes into typed trade
constraints; a deterministic pre-trade checker enforces them.

## Rules
- uv for everything Python: uv add / uv sync / uv run. Never pip, venv, conda, or bare python/python3.
- data/gold and data/test are read-only: never edit, never train on them.
- Every reported metric must come from src/fineprint/eval. Never estimate numbers.
- Headline metrics are deterministic; no LLM-as-judge.
- Fix seeds; log config, data hash and adapter SHA for every run.
- The model never decides allow/block; the checker does.
- Local machine has no GPU; train/serve deps live in extras.
