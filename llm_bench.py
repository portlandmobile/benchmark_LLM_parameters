#!/usr/bin/env python3
"""Call an LLM (Claude or Gemini) with arbitrary add-on parameters and record what comes back.

Examples:
    python llm_bench.py --provider claude --prompt "Explain quicksort" --param output_config.effort=low
    python llm_bench.py --provider gemini --prompts-file prompts.yaml --prompt-id 2 \
        --param thinking_config.thinking_level=high
    python llm_bench.py --provider claude --list-prompts --prompts-file prompts.yaml
    python llm_bench.py --provider gemini --list-models
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

MAX_PROMPTS = 10
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_MODELS = {
    "claude": "claude-opus-5",
    "gemini": "gemini-3.1-pro-preview",
}

# Defaults applied before user params. Any of these can be overridden with --param,
# or removed entirely with --param <key>=null.
DEFAULT_PARAMS = {
    "claude": {
        "max_tokens": 64000,
        # Summarized thinking so the reasoning is visible (the API default returns empty thinking text).
        "thinking": {"type": "adaptive", "display": "summarized"},
    },
    "gemini": {
        "thinking_config": {"include_thoughts": True},
    },
}


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

def parse_value(raw: str) -> Any:
    """Parse a CLI value as JSON when possible (numbers, bools, null, objects), else keep as string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def set_dotted(target: dict, dotted_key: str, value: Any) -> None:
    """Set a.b.c=value inside target, creating nested dicts as needed."""
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def deep_merge(base: dict, override: dict, keep_none: bool = False) -> dict:
    """Recursively merge override into a copy of base.

    None values in override delete the key, unless keep_none is set (used while collecting
    user layers, so the deletion marker survives until it is applied to the defaults).
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if value is None and not keep_none:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value, keep_none)
        else:
            result[key] = copy.deepcopy(value)
    return result


def strip_none(value: Any) -> Any:
    """Drop leftover None deletion markers (ones that had no default to remove)."""
    if isinstance(value, dict):
        return {k: strip_none(v) for k, v in value.items() if v is not None}
    return value


def load_structured_file(path: str) -> Any:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def build_params(args: argparse.Namespace) -> tuple[dict, dict]:
    """Return (user_params, final_params) where final = defaults <- params file <- --params-json <- --param."""
    user: dict = {}
    if args.params_file:
        user = deep_merge(user, load_structured_file(args.params_file) or {}, keep_none=True)
    if args.params_json:
        user = deep_merge(user, json.loads(args.params_json), keep_none=True)
    for item in args.param or []:
        if "=" not in item:
            sys.exit(f"error: --param must be key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(user, key.strip(), parse_value(raw))

    defaults = {} if args.no_defaults else DEFAULT_PARAMS[args.provider]
    return user, strip_none(deep_merge(defaults, user))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> list[dict]:
    """Load prompts from YAML/JSON. Accepts a list, or {prompts: [...]}; items are strings or {id, name, text}."""
    data = load_structured_file(path)
    if isinstance(data, dict):
        data = data.get("prompts", [])
    if not isinstance(data, list):
        sys.exit(f"error: {path} must contain a list of prompts")
    if len(data) > MAX_PROMPTS:
        sys.exit(f"error: {path} has {len(data)} prompts; the limit is {MAX_PROMPTS}")

    prompts = []
    for i, item in enumerate(data, start=1):
        if isinstance(item, str):
            item = {"text": item}
        prompts.append({
            "id": str(item.get("id", i)),
            "name": item.get("name", ""),
            "text": item["text"],
            "system": item.get("system"),
        })
    return prompts


def resolve_prompt(args: argparse.Namespace) -> dict:
    if args.prompt is not None:
        text = sys.stdin.read() if args.prompt == "-" else args.prompt
        return {"id": None, "name": "inline", "text": text, "system": None}
    if not args.prompts_file:
        sys.exit("error: provide --prompt \"...\" or --prompts-file FILE --prompt-id ID")
    prompts = load_prompts(args.prompts_file)
    if args.prompt_id is None:
        if len(prompts) == 1:
            return prompts[0]
        sys.exit(f"error: {args.prompts_file} has {len(prompts)} prompts; choose one with --prompt-id")
    for p in prompts:
        if p["id"] == str(args.prompt_id):
            return p
    sys.exit(f"error: no prompt with id {args.prompt_id!r} in {args.prompts_file}")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass
class Result:
    response_text: str = ""
    thinking_text: str = ""
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)
    raw_response: Any = None


def call_claude(model: str, prompt: str, system: str | None, params: dict) -> Result:
    import anthropic

    client = anthropic.Anthropic()
    params = copy.deepcopy(params)
    betas = params.pop("betas", None)
    timeout = params.pop("timeout", None)

    request = {"model": model, "messages": [{"role": "user", "content": prompt}], **params}
    if system:
        request["system"] = system

    # Every param goes through extra_body so any field the API accepts works,
    # even ones this SDK version doesn't know about yet.
    max_tokens = request.pop("max_tokens", 64000)
    body_extras = {k: v for k, v in request.items() if k not in ("model", "messages")}
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": request["messages"],
        "extra_body": body_extras,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    messages_api = client.beta.messages if betas else client.messages
    if betas:
        kwargs["betas"] = betas

    # Streaming avoids HTTP timeouts on long thinking runs; we only need the final message.
    with messages_api.stream(**kwargs) as stream:
        message = stream.get_final_message()

    thinking, text = [], []
    for block in message.content:
        if block.type == "thinking":
            thinking.append(block.thinking)
        elif block.type == "redacted_thinking":
            thinking.append("[redacted thinking]")
        elif block.type == "text":
            text.append(block.text)

    raw = message.model_dump(mode="json")
    if message.stop_reason == "refusal" and getattr(message, "stop_details", None):
        text.append(f"[refusal: {message.stop_details}]")

    return Result(
        response_text="\n".join(text),
        thinking_text="\n\n".join(t for t in thinking if t),
        stop_reason=message.stop_reason,
        usage=raw.get("usage") or {},
        request={**request, "max_tokens": max_tokens, **({"betas": betas} if betas else {})},
        raw_response=raw,
    )


def call_gemini(model: str, prompt: str, system: str | None, params: dict) -> Result:
    from google import genai

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
    config = copy.deepcopy(params)
    if system:
        config["system_instruction"] = system

    # config is passed as a dict: any GenerateContentConfig field works, and fields the SDK
    # doesn't model yet can be sent raw via http_options.extra_body.<field>=...
    response = client.models.generate_content(model=model, contents=prompt, config=config or None)

    thinking, text = [], []
    finish_reason = None
    for candidate in response.candidates or []:
        finish_reason = str(candidate.finish_reason.name if candidate.finish_reason else None)
        for part in (candidate.content.parts if candidate.content else None) or []:
            if not part.text:
                continue
            (thinking if part.thought else text).append(part.text)
        break  # first candidate only

    raw = response.model_dump(mode="json", exclude_none=True)
    if not text and response.prompt_feedback:
        text.append(f"[blocked: {response.prompt_feedback}]")

    return Result(
        response_text="".join(text),
        thinking_text="".join(thinking),
        stop_reason=finish_reason,
        usage=raw.get("usage_metadata") or {},
        request={"model": model, "contents": prompt, "config": config},
        raw_response=raw,
    )


PROVIDERS = {"claude": call_claude, "gemini": call_gemini}


def list_models(provider: str) -> None:
    if provider == "claude":
        import anthropic
        for m in anthropic.Anthropic().models.list():
            print(f"{m.id:40} {m.display_name}")
    else:
        from google import genai
        for m in genai.Client().models.list():
            if "generateContent" in (m.supported_actions or []):
                print(f"{m.name.removeprefix('models/'):40} {m.display_name or ''}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def token_summary(provider: str, usage: dict) -> str:
    if provider == "claude":
        parts = [f"input={usage.get('input_tokens')}", f"output={usage.get('output_tokens')}"]
        if usage.get("cache_read_input_tokens"):
            parts.append(f"cache_read={usage['cache_read_input_tokens']}")
        return " ".join(parts) + "  (output includes thinking)"
    return (
        f"input={usage.get('prompt_token_count')} output={usage.get('candidates_token_count')} "
        f"thinking={usage.get('thoughts_token_count')} total={usage.get('total_token_count')}"
    )


def print_report(record: dict, show_thinking: bool) -> None:
    rule = "=" * 78
    print(rule)
    print(f"{record['provider']} / {record['model']}   latency {record['latency_seconds']:.2f}s   "
          f"stop={record['stop_reason']}")
    print(f"params: {json.dumps(record['params_user']) if record['params_user'] else '(defaults only)'}")
    print(f"tokens: {token_summary(record['provider'], record['usage'])}")
    if show_thinking and record["thinking_text"]:
        print(rule)
        print("THINKING:\n")
        print(record["thinking_text"])
    print(rule)
    print("RESPONSE:\n")
    print(record["response_text"])
    print(rule)


def save_record(record: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pid = f"p{record['prompt']['id']}" if record["prompt"]["id"] else "inline"
    safe_model = record["model"].replace("/", "_")
    path = out_dir / f"{stamp}_{record['provider']}_{safe_model}_{pid}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test how an LLM behaves with different parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--provider", "-p", required=True, choices=sorted(PROVIDERS))
    p.add_argument("--model", "-m", help="model ID (default: claude=%(c)s, gemini=%(g)s)"
                   % {"c": DEFAULT_MODELS["claude"], "g": DEFAULT_MODELS["gemini"]})

    src = p.add_argument_group("prompt (inline or from file)")
    src.add_argument("--prompt", help='prompt text; use "-" to read from stdin')
    src.add_argument("--prompts-file", help=f"YAML/JSON file with up to {MAX_PROMPTS} prompts")
    src.add_argument("--prompt-id", help="which prompt in --prompts-file to run")
    src.add_argument("--system", help="system prompt (overrides one set in the prompts file)")

    prm = p.add_argument_group("add-on parameters (merged in this order; later wins)")
    prm.add_argument("--params-file", help="YAML/JSON file of parameters (a reusable preset)")
    prm.add_argument("--params-json", help="parameters as a JSON object string")
    prm.add_argument("--param", action="append", metavar="KEY=VALUE",
                     help="one parameter; dotted keys nest (output_config.effort=high); values parse as "
                          "JSON when possible; VALUE=null removes a key. Repeatable.")
    prm.add_argument("--no-defaults", action="store_true",
                     help="don't apply the tool's built-in default params; send only what you pass")

    out = p.add_argument_group("output")
    out.add_argument("--hide-thinking", action="store_true", help="don't print thinking text")
    out.add_argument("--out-dir", default=str(RESULTS_DIR), help="where JSON results are saved")
    out.add_argument("--no-save", action="store_true", help="don't write a JSON result file")
    out.add_argument("--dry-run", action="store_true", help="print the request without calling the API")

    p.add_argument("--list-prompts", action="store_true", help="list prompts in --prompts-file and exit")
    p.add_argument("--list-models", action="store_true", help="list available models for --provider and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    if args.list_models:
        list_models(args.provider)
        return 0
    if args.list_prompts:
        if not args.prompts_file:
            sys.exit("error: --list-prompts needs --prompts-file")
        for pr in load_prompts(args.prompts_file):
            first_line = pr["text"].strip().splitlines()[0] if pr["text"].strip() else ""
            print(f"[{pr['id']}] {pr['name'] or ''}  {first_line[:70]}")
        return 0

    prompt = resolve_prompt(args)
    system = args.system or prompt.get("system")
    model = args.model or DEFAULT_MODELS[args.provider]
    params_user, params_final = build_params(args)

    if args.dry_run:
        print(json.dumps({"provider": args.provider, "model": model, "system": system,
                          "prompt": prompt["text"], "params": params_final}, indent=2))
        return 0

    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": args.provider,
        "model": model,
        "prompt": prompt,
        "system": system,
        "params_user": params_user,
        "params_sent": params_final,
    }

    started = time.perf_counter()
    try:
        result = PROVIDERS[args.provider](model, prompt["text"], system, params_final)
        error = None
    except Exception as exc:  # record API errors (e.g. a rejected parameter) as a result too
        result, error = Result(), f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started

    record.update({
        "latency_seconds": round(latency, 3),
        "error": error,
        **{k: v for k, v in asdict(result).items() if k != "raw_response"},
        "raw_response": result.raw_response,
    })

    if error:
        print(f"ERROR after {latency:.2f}s: {error}", file=sys.stderr)
    else:
        print_report(record, show_thinking=not args.hide_thinking)

    if not args.no_save:
        path = save_record(record, Path(args.out_dir))
        print(f"saved: {path}")
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
