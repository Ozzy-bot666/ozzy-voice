# Ozzy Voice

OpenAI-compatible endpoint for voice conversations with Ozzy (powered by Claude Sonnet).

Works with:
- **Retell AI** (Custom LLM)
- **Pipecat** (OpenAI-compatible LLM)
- Any platform supporting OpenAI's `/v1/chat/completions` API

## Features

- 🎯 OpenAI-compatible API (`/v1/chat/completions`)
- 🧠 Automatic memory injection (SOUL.md, USER.md, MEMORY.md)
- ⚡ Streaming support (essential for low-latency voice)
- 🔒 Optional API key authentication
- 🦇 Same Ozzy personality, faster responses

## Setup

```bash
# Install dependencies
npm install

# Copy and edit environment
cp .env.example .env

# Run
npm start
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Your Anthropic API key |
| `MODEL` | No | `claude-sonnet-4-20250514` | Claude model to use |
| `MEMORY_PATH` | No | `/home/node/clawd` | Path to memory files |
| `API_KEY` | No | - | Optional auth key |
| `PORT` | No | `3000` | Server port |

## API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions.

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "messages": [{"role": "user", "content": "Hey Ozzy!"}],
    "stream": true
  }'
```

### `GET /v1/models`

List available models.

### `GET /health`

Health check endpoint.

## Retell Configuration

1. Create a Custom LLM in Retell
2. Set the endpoint URL to `https://your-deployment.onrender.com/v1/chat/completions`
3. Add your API key as Bearer token (if configured)
4. Select "OpenAI Compatible" as the format

## Pipecat Configuration

```python
from pipecat.services.openai import OpenAILLMService

llm = OpenAILLMService(
    api_key="your-api-key",
    base_url="https://your-deployment.onrender.com/v1",
    model="claude-sonnet-4-20250514"
)
```

## Memory Files

The endpoint automatically reads and injects:

- `SOUL.md` - Ozzy's personality
- `USER.md` - User information
- `MEMORY.md` - Long-term memory
- `memory/YYYY-MM-DD.md` - Today's context

These are injected into the system prompt for every request.

## Deployment on Render

1. Push to GitHub
2. Create new Web Service on Render
3. Connect your repo
4. Add environment variables
5. Deploy

For shared memory with the main Clawdbot instance, you'll need either:
- A persistent disk mounted at `/home/node/clawd`
- External storage (S3, etc.) synced to both services
- Or accept that voice-Ozzy has read-only snapshot of memory at deploy time
