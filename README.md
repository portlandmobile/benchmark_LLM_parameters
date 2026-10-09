# LLM parameter bench

Send one prompt to Claude or Gemini with whatever parameters you want, and see what comes back:
response, thinking, token usage, latency, and stop reason. Every run is also saved as JSON under `results/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in ANTHROPIC_API_KEY and/or GEMINI_API_KEY
source .venv/bin/activate
```

## Usage

```bash
# Inline prompt
python llm_bench.py -p claude --prompt "Explain quicksort" --param output_config.effort=low

# Prompt from file (up to 10 prompts), picked by id
python llm_bench.py -p gemini --prompts-file prompts.yaml --prompt-id 2 \
    --param thinking_config.thinking_level=high
python llm_bench.py -p claude --prompts-file prompts.yaml --list-prompts

# Other model, preset file, see the request without calling
python llm_bench.py -p claude -m claude-sonnet-5 --params-file presets/claude-max-effort.yaml --prompt "..." --dry-run

# What models can I use?
python llm_bench.py -p gemini --list-models
```

## Parameters

Parameters are passed through to the API **as-is**, using each provider's own names. Nothing is
validated locally, so new or unusual parameters work. If the API rejects one, the error is shown and saved.

| Way to pass | Example |
|---|---|
| `--param KEY=VALUE` (repeatable) | `--param output_config.effort=high` |
| dotted keys nest | `thinking_config.thinking_level=low` → `{"thinking_config": {"thinking_level": "low"}}` |
| values are JSON when they parse | `--param temperature=0.2`, `--param 'stop_sequences=["END"]'` |
| `VALUE=null` removes a key | `--param thinking=null` |
| `--params-json '{...}'` | `--params-json '{"output_config": {"effort": "max"}}'` |
| `--params-file preset.yaml` | see `presets/` |

Merge order (later wins): built-in defaults → `--params-file` → `--params-json` → `--param`.
Use `--no-defaults` to send only what you pass.

**Built-in defaults**
- Claude: `max_tokens=64000`, `thinking={"type": "adaptive", "display": "summarized"}`, so the reasoning summary is visible.
- Gemini: `thinking_config.include_thoughts=true`, so thought summaries are returned.

**Provider notes**
- **Claude:** parameters go into the Messages API request body. `betas=[...]` is sent as the `anthropic-beta` header,
  and `timeout` is a client option. Common knobs: `output_config.effort` (`low`…`max`), `thinking`, `max_tokens`.
- **Gemini:** parameters go into `GenerateContentConfig`. Common knobs: `thinking_config.thinking_level`,
  `thinking_config.thinking_budget`, `temperature`, `max_output_tokens`. To send fields the SDK doesn't know about, use
  `http_options.extra_body.<field>=...`.

## Prompts file

```yaml
prompts:
  - id: 1
    name: optional-label
    system: optional system prompt
    text: |
      The prompt text.
```
A plain list of strings also works. `--system` on the command line overrides a prompt's `system`.
