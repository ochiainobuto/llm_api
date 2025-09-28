# handler.py
import os
import logging
import runpod

logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL = os.getenv("MODEL", "gemini-2.5-flash-lite")

# --- Provider 判定 ---
def _detect_provider(model_name: str, explicit: str | None) -> str:
    if explicit:
        p = explicit.strip().lower()
        if p in ("openai", "gemini"):
            return p
    name = (model_name or "").lower()
    if name.startswith(("gpt-", "gpt4", "gpt-4", "gpt-4o", "o3", "o4", "gpt-4.1")) or name.startswith("openai:"):
        return "openai"
    if name.startswith(("gemini-", "models/gemini-")) or name.startswith("gemini:"):
        return "gemini"
    return "gemini"

# --- 入力ヘルパ ---
def _get_prompt(payload: dict) -> str:
    return (
        payload.get("prompt")
        or payload.get("text")
        or payload.get("query")
        or payload.get("message")
        or ""
    )

def _number(x, default):
    try:
        return type(default)(x)
    except Exception:
        return default

# --- Gemini（遅延初期化＆モデルキャッシュ） ---
_genai = None
_gemini_models = {}

def _gemini_get_model(model_name: str):
    global _genai
    if _genai is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY が未設定です")
        import google.generativeai as genai  # 遅延 import
        genai.configure(api_key=api_key)
        _genai = genai
        logging.info("Gemini client initialized.")
    if model_name not in _gemini_models:
        _gemini_models[model_name] = _genai.GenerativeModel(model_name)
    return _gemini_models[model_name]

# --- OpenAI（遅延初期化）---
_oa_client = None

def _openai_client():
    global _oa_client
    if _oa_client is not None:
        return _oa_client
    from openai import OpenAI  # 遅延 import
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です")
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    _oa_client = OpenAI(api_key=api_key, base_url=base) if base else OpenAI(api_key=api_key)
    logging.info(f"OpenAI client initialized. base={'custom' if base else 'official'}")
    return _oa_client

def handler(job):
    try:
        payload = (job or {}).get("input") or {}

        model_name  = payload.get("model") or DEFAULT_MODEL
        provider    = _detect_provider(model_name, payload.get("provider"))
        temperature = _number(payload.get("temperature", 0.7), 0.7)
        max_tokens  = _number(payload.get("max_tokens", 256), 256)
        top_p       = _number(payload.get("top_p", 1.0), 1.0)
        stop        = payload.get("stop")

        # できるだけ最短経路で生成して返す
        if provider == "openai":
            client = _openai_client()
            messages = payload.get("messages")
            if not messages:
                prompt = _get_prompt(payload)
                if not prompt:
                    return {"ok": False, "error": "プロンプトがありません"}
                messages = [{"role": "user", "content": prompt}]
            # Chat Completions（広く互換・低オーバーヘッド）
            res = client.chat.completions.create(
                model=model_name.replace("openai:", ""),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop
            )
            text = res.choices[0].message.content if res.choices else ""

        else:
            # Gemini は単発プロンプト最短
            prompt = _get_prompt(payload)
            if not prompt:
                # messages が来たら user だけ連結（最短処理）
                msgs = payload.get("messages") or []
                prompt = "\n".join([m.get("content", "") for m in msgs if m.get("role") == "user"]).strip()
                if not prompt:
                    return {"ok": False, "error": "プロンプトがありません"}
            model = _gemini_get_model(model_name.replace("gemini:", ""))
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens),
                    "top_p": float(top_p),
                    **({"stop_sequences": stop} if stop else {})
                },
            )
            text = getattr(resp, "text", "") or ""

        # 生テキストで即返し（runsync の output が文字列になる）
        if str(payload.get("action", "")).lower() == "ask" or payload.get("raw") is True:
            return text

        # JSON で返すモード
        return {
            "ok": True,
            "status": "success",
            "provider": provider,
            "model": model_name,
            "response": text,
        }

    except Exception as e:
        logging.exception("handler failed")
        return {"ok": False, "status": "error", "error": str(e)}

# RunPod serverless
runpod.serverless.start({"handler": handler})
