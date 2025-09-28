# handler.py
import os
import json
import logging
import runpod

logging.basicConfig(level=logging.INFO)

# 既定モデル（Env で上書き可）
DEFAULT_MODEL = os.getenv("MODEL", "gemini-2.5-flash-lite")

# モデルのキャッシュ（遅延初期化用）
_genai = None
_model = None
_model_name = None

def _get_prompt(payload: dict) -> str:
    # いろいろなキー名に対応
    return (
        payload.get("prompt")
        or payload.get("text")
        or payload.get("query")
        or payload.get("message")
        or ""
    )

def _ensure_model(model_name: str):
    """必要になった時点で一度だけ Gemini を初期化してモデルをキャッシュ。"""
    global _genai, _model, _model_name

    if _model is not None and _model_name == model_name:
        return _genai, _model

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        # 初期化時に落とさず、呼び出し側にエラーを返す
        raise RuntimeError("GEMINI_API_KEY が未設定です（Endpoint > Environment Variables に設定してください）")

    import google.generativeai as genai  # ← ここで初めて import（起動時クラッシュ回避）
    genai.configure(api_key=api_key)

    _genai = genai
    _model = genai.GenerativeModel(model_name)
    _model_name = model_name
    logging.info(f"Gemini model initialized: {model_name}")
    return _genai, _model

def handler(job):
    """
    RunPod Serverless handler
    """
    try:
        payload = (job or {}).get("input") or {}
        prompt = _get_prompt(payload)
        if not prompt:
            return {"ok": False, "error": "プロンプトが提供されていません"}

        # 入力から可変パラメータ取得（なければ既定）
        model_name   = payload.get("model") or DEFAULT_MODEL
        temperature  = float(payload.get("temperature", 0.7))
        max_tokens   = int(payload.get("max_tokens", 256))  # まずは短めに

        genai, model = _ensure_model(model_name)

        # 生成
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        text = getattr(resp, "text", None) or ""
        return {
            "ok": True,
            "status": "success",
            "model": model_name,
            "prompt": prompt,
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
