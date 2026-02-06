import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const app = express();
app.use(express.json());

// Config
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.MODEL || 'claude-sonnet-4-20250514';
const MEMORY_PATH = process.env.MEMORY_PATH || '/home/node/clawd';
const API_KEY = process.env.API_KEY; // Optional: protect the endpoint

// Initialize Anthropic client
const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

// Read memory files and build system context
function buildSystemContext() {
  const files = ['SOUL.md', 'USER.md', 'MEMORY.md'];
  let context = `You are Ozzy, a snarky AI bat assistant. You're currently in a voice conversation.
Keep responses concise and conversational - this is spoken, not written.
Be natural, use contractions, and don't be overly formal.

`;

  for (const file of files) {
    const path = join(MEMORY_PATH, file);
    if (existsSync(path)) {
      try {
        const content = readFileSync(path, 'utf-8');
        context += `\n## ${file}\n${content}\n`;
      } catch (e) {
        console.error(`Failed to read ${file}:`, e.message);
      }
    }
  }

  // Also try to read today's memory
  const today = new Date().toISOString().split('T')[0];
  const todayPath = join(MEMORY_PATH, 'memory', `${today}.md`);
  if (existsSync(todayPath)) {
    try {
      const content = readFileSync(todayPath, 'utf-8');
      context += `\n## Today's Context (${today})\n${content}\n`;
    } catch (e) {
      console.error(`Failed to read today's memory:`, e.message);
    }
  }

  return context;
}

// Convert OpenAI messages to Anthropic format
function convertMessages(openaiMessages) {
  const messages = [];
  let systemPrompt = '';

  for (const msg of openaiMessages) {
    if (msg.role === 'system') {
      systemPrompt += msg.content + '\n';
    } else if (msg.role === 'user' || msg.role === 'assistant') {
      messages.push({
        role: msg.role,
        content: msg.content
      });
    }
  }

  return { systemPrompt, messages };
}

// Convert Anthropic response to OpenAI format
function toOpenAIResponse(anthropicResponse, model) {
  const content = anthropicResponse.content[0]?.text || '';
  
  return {
    id: `chatcmpl-${anthropicResponse.id}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: content
      },
      finish_reason: anthropicResponse.stop_reason === 'end_turn' ? 'stop' : anthropicResponse.stop_reason
    }],
    usage: {
      prompt_tokens: anthropicResponse.usage?.input_tokens || 0,
      completion_tokens: anthropicResponse.usage?.output_tokens || 0,
      total_tokens: (anthropicResponse.usage?.input_tokens || 0) + (anthropicResponse.usage?.output_tokens || 0)
    }
  };
}

// Convert Anthropic stream events to OpenAI SSE format
function toOpenAIStreamChunk(event, model) {
  if (event.type === 'content_block_delta' && event.delta?.text) {
    return {
      id: `chatcmpl-stream`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        delta: {
          content: event.delta.text
        },
        finish_reason: null
      }]
    };
  } else if (event.type === 'message_stop') {
    return {
      id: `chatcmpl-stream`,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: model,
      choices: [{
        index: 0,
        delta: {},
        finish_reason: 'stop'
      }]
    };
  }
  return null;
}

// Auth middleware
function authenticate(req, res, next) {
  if (!API_KEY) return next(); // No auth configured
  
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }
  
  const token = authHeader.substring(7);
  if (token !== API_KEY) {
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  next();
}

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', model: MODEL });
});

// OpenAI-compatible chat completions endpoint
app.post('/v1/chat/completions', authenticate, async (req, res) => {
  try {
    const { messages, stream = false, max_tokens = 1024, temperature = 0.7 } = req.body;
    
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'messages array is required' });
    }

    // Build context from memory files
    const memoryContext = buildSystemContext();
    
    // Convert messages
    const { systemPrompt, messages: anthropicMessages } = convertMessages(messages);
    const fullSystemPrompt = memoryContext + '\n' + systemPrompt;

    // Ensure we have at least one message
    if (anthropicMessages.length === 0) {
      anthropicMessages.push({ role: 'user', content: 'Hello' });
    }

    if (stream) {
      // Streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const response = await anthropic.messages.stream({
        model: MODEL,
        max_tokens: max_tokens,
        temperature: temperature,
        system: fullSystemPrompt,
        messages: anthropicMessages
      });

      for await (const event of response) {
        const chunk = toOpenAIStreamChunk(event, MODEL);
        if (chunk) {
          res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        }
      }
      
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      // Non-streaming response
      const response = await anthropic.messages.create({
        model: MODEL,
        max_tokens: max_tokens,
        temperature: temperature,
        system: fullSystemPrompt,
        messages: anthropicMessages
      });

      res.json(toOpenAIResponse(response, MODEL));
    }
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      error: { 
        message: error.message, 
        type: 'api_error' 
      } 
    });
  }
});

// Models endpoint (for compatibility)
app.get('/v1/models', authenticate, (req, res) => {
  res.json({
    object: 'list',
    data: [{
      id: MODEL,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'anthropic'
    }]
  });
});

app.listen(PORT, () => {
  console.log(`Ozzy Voice endpoint running on port ${PORT}`);
  console.log(`Model: ${MODEL}`);
  console.log(`Memory path: ${MEMORY_PATH}`);
  console.log(`Auth: ${API_KEY ? 'enabled' : 'disabled'}`);
});
