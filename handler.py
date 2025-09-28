# handler.py
import os
import json
import logging
import runpod

logging.basicConfig(level=logging.INFO)

# 既定モデル（Env で上書き可）
DEFAULT_MODEL = os.getenv("MODEL", "gemini-2.5-flash-lite")

# ====== Provider 判定ロジック ======
OPENAI_PREFIXES = (
    "gpt-", "gpt4", "gpt-4", "gpt-4o", "o3", "o4", "o-mini", "gpt-4.1", "gpt-4.1-mini"
)
GEMINI_PREFIXES = ("gemini-", "models/gemini-")

def _detect_provider(model_name: str, explicit: str | None) -> str:
    """
    provider 明示指定があればそれを優先（'openai' or 'gemini'）。
    無ければ model 名のプレフィクスで推定。
    どちらでもなければ gemini を既定。
    """
    if explicit:
        p = explicit.strip().lower()
        if p in ("openai", "gemini"):
            return p
    name = (model_name or "").lower()
    if name.startswith(OPENAI_PREFIXES) or name.startswith("openai:"):
        return "openai"
    if name.startswith(GEMINI_PREFIXES) or name.startswith("gemini:"):
        return "gemini"
    return "gemini"

# ====== Gemini（遅延初期化） ======
_genai = None
_gemini_model_cache = {}
def _get_gemini_model(model_name: str):
    global _genai
    if _genai is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY が未設定です（Endpoint > Environment Variables に設定してください）")
        import google.generativeai as genai  # 起動時クラッシュ回避のため遅延 import
        genai.configure(api_key=api_key)
        _genai = genai
        logging.info("Gemini client initialized.")
    if model_name not in _gemini_model_cache:
        _gemini_model_cache[model_name] = _genai.GenerativeModel(model_name)
        logging.info(f"Gemini model cached: {model_name}")
    return _gemini_model_cache[model_name]

# ====== OpenAI（遅延初期化） ======
_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です（Endpoint > Environment Variables に設定してください）")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
    if base_url:
        _openai_client = OpenAI(api_key=api_key, base_url=base_url)
        logging.info(f"OpenAI client initialized with base_url: {base_url}")
    else:
        _openai_client = OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized.")
    return _openai_client

# ====== 入力ユーティリティ ======
def _get_prompt(payload: dict) -> str:
    # いろいろなキー名に対応
    return (
        payload.get("prompt")
        or payload.get("text")
        or payload.get("query")
        or payload.get("message")
        or ""
    )

def _get_messages(payload: dict):
    """
    OpenAI 'messages' 互換の配列が来たら優先採用。
    形式：
      [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
    """
    msgs = payload.get("messages")
    if isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs):
        return msgs
    # なければ prompt を user 発話に変換
    prompt = _get_prompt(payload)
    system = payload.get("system") or ""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": str(system)})
    msgs.append({"role": "user", "content": str(prompt)})
    return msgs

def _number(x, default):
    try:
        return type(default)(x)
    except Exception:
        return default

# ====== メイン handler ======
def handler(job):
    """
    RunPod Serverless handler
    - provider: "gemini" | "openai"（省略可、model から推論）
    - model:    例) "gemini-2.5-flash-lite", "gpt-4o-mini", "o3-mini"
    - messages: OpenAI 互換配列（優先）
    - prompt/text/query/message: 単発プロンプト（messages 無い場合に使用）
    - temperature, max_tokens, top_p, top_k など任意
    """
    try:
        payload = (job or {}).get("input") or {}
        model_name   = payload.get("model") or DEFAULT_MODEL
        provider     = _detect_provider(model_name, payload.get("provider"))
        temperature  = _number(payload.get("temperature", 0.7), 0.7)
        max_tokens   = _number(payload.get("max_tokens", 512), 512)
        top_p        = _number(payload.get("top_p", 1.0), 1.0)

        # OpenAI / Gemini 共通メッセージ整形
        messages = _get_messages(payload)

        if provider == "openai":
            client = _get_openai_client()
            # Responses API か Chat Completions かを切り替えるならここで条件分岐可。
            # 汎用性のため Chat Completions を採用（広く互換あり）。
            completion = client.chat.completions.create(
                model=model_name.replace("openai:", ""),  # "openai:" プレフィクスを外しておく
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            text = completion.choices[0].message.content if completion.choices else ""
            return {
                "ok": True,
                "status": "success",
                "provider": provider,
                "model": model_name,
                "messages": messages,
                "response": text,
            }

        # === Gemini ===
        # Gemini は messages を 1 本のプロンプトに結合する簡易方式
        # （必要に応じて ChatSession へ拡張可能）
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_parts = [m["content"] for m in messages if m.get("role") in ("user", "assistant")]
        combined_prompt = ""
        if sys_parts:
            combined_prompt += "System:\n" + "\n".join(map(str, sys_parts)) + "\n\n"
        combined_prompt += "User:\n" + "\n\n".join(map(str, user_parts))

        model = _get_gemini_model(model_name.replace("gemini:", ""))
        resp = model.generate_content(
            combined_prompt,
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(max_tokens),
                "top_p": float(top_p),
            },
        )
        text = getattr(resp, "text", None) or ""
        return {
            "ok": True,
            "status": "success",
            "provider": provider,
            "model": model_name,
            "messages": messages,
            "response": text,
        }

    except Exception as e:
        logging.exception("handler failed")
        return {
            "ok": False,
            "status": "error",
            "error": str(e),
        }

# サーバレス起動
runpod.serverless.start({"handler": handler})
