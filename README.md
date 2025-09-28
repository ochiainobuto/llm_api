# serverless-text-gen (Gemini / OpenAI 対応)

RunPod Serverless のエンドポイントで、**Gemini** と **OpenAI** の両方を選択的に使える最小実装。

## 環境変数（RunPod Endpoint > Environment Variables）

- `MODEL`（任意）: 既定モデル。例: `gemini-2.5-flash-lite` / `gpt-4o-mini`
- `GEMINI_API_KEY`: Gemini の API キー
- `OPENAI_API_KEY`: OpenAI の API キー
- `OPENAI_BASE_URL`（任意）: 自前ゲートウェイ等を使う場合に指定

## 呼び出し形式

### 1) 単発プロンプト（Gemini）
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{
    "input": {
      "provider": "gemini",
      "model": "gemini-2.5-flash-lite",
      "prompt": "こんにちは！要約して"
    }
  }'
