import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { readFileSync, existsSync, appendFileSync } from 'fs';
import { join } from 'path';

const app = express();

// Security middleware
app.use(helmet());
app.use(express.json({ limit: '100kb' }));

// Config
const PORT = process.env.PORT || 3000;
const PROVIDER = process.env.PROVIDER || 'anthropic'; // anthropic | google | openai
const MODEL = process.env.MODEL || getDefaultModel(PROVIDER);
const MEMORY_PATH = process.env.MEMORY_PATH || '/home/node/clawd';
const API_KEY = process.env.API_KEY;
const ALLOWED_IPS = process.env.ALLOWED_IPS?.split(',').map(ip => ip.trim()).filter(Boolean) || [];
const LOG_REQUESTS = process.env.LOG_REQUESTS === 'true';

// Provider-specific config
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

function getDefaultModel(provider) {
  switch (provider) {
    case 'anthropic': return 'claude-sonnet-4-20250514';
    case 'google': return 'gemini-2.0-flash';
    case 'openai': return 'gpt-4o-mini';
    default: return 'claude-sonnet-4-20250514';
  }
}

// Initialize clients
let anthropic, google, openai;

if (ANTHROPIC_API_KEY) {
  anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
}
if (GOOGLE_API_KEY) {
  google = new GoogleGenerativeAI(GOOGLE_API_KEY);
}
if (OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: OPENAI_API_KEY });
}

// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
  message: { error: 'Too many requests, please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip
});

app.use('/v1', limiter);

// Audit logging
function auditLog(req, status, details = {}) {
  if (!LOG_REQUESTS) return;
  
  const entry = {
    timestamp: new Date().toISOString(),
    ip: req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip,
    method: req.method,
    path: req.path,
    provider: PROVIDER,
    model: MODEL,
    status,
    ...details
  };
  
  console.log('[AUDIT]', JSON.stringify(entry));
}

// IP whitelist middleware
function ipWhitelist(req, res, next) {
  if (ALLOWED_IPS.length === 0) return next();
  
  const clientIp = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip;
  
  const allowed = ALLOWED_IPS.some(allowedIp => {
    if (allowedIp === '*') return true;
    if (allowedIp.endsWith('*')) {
      return clientIp.startsWith(allowedIp.slice(0, -1));
    }
    return clientIp === allowedIp;
  });
  
  if (!allowed) {
    auditLog(req, 403, { reason: 'IP not whitelisted', clientIp });
    return res.status(403).json({ error: 'Forbidden' });
  }
  
  next();
}

// Auth middleware
function authenticate(req, res, next) {
  if (!API_KEY) return next();
  
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    auditLog(req, 401, { reason: 'Missing auth header' });
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }
  
  if (authHeader.substring(7) !== API_KEY) {
    auditLog(req, 401, { reason: 'Invalid API key' });
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  next();
}

// Read memory files and build system context
function buildSystemContext() {
  const files = ['SOUL.md', 'USER.md', 'MEMORY.md'];
  let context = `You are Ozzy, a snarky AI bat assistant. You're currently in a voice conversation.
Keep responses concise and conversational - this is spoken, not written.
Be natural, use contractions, and don't be overly formal.
Respond in the same language as the user speaks to you.

`;

  for (const file of files) {
    const path = join(MEMORY_PATH, file);
    if (existsSync(path)) {
      try {
        context += `\n## ${file}\n${readFileSync(path, 'utf-8')}\n`;
      } catch (e) {
        console.error(`Failed to read ${file}:`, e.message);
      }
    }
  }

  const today = new Date().toISOString().split('T')[0];
  const todayPath = join(MEMORY_PATH, 'memory', `${today}.md`);
  if (existsSync(todayPath)) {
    try {
      context += `\n## Today's Context (${today})\n${readFileSync(todayPath, 'utf-8')}\n`;
    } catch (e) {
      console.error(`Failed to read today's memory:`, e.message);
    }
  }

  return context;
}

// Validate request
function validateRequest(body) {
  if (!body?.messages || !Array.isArray(body.messages)) return 'messages array is required';
  if (body.messages.length > 100) return 'Too many messages (max 100)';
  
  for (const msg of body.messages) {
    if (!['system', 'user', 'assistant'].includes(msg.role)) return 'Invalid message role';
    if (msg.content === undefined || msg.content === null) return 'Message content is required';
  }
  
  if (body.max_tokens && (body.max_tokens < 1 || body.max_tokens > 8192)) return 'max_tokens must be between 1 and 8192';
  if (body.temperature && (body.temperature < 0 || body.temperature > 2)) return 'temperature must be between 0 and 2';
  
  return null;
}

// ============ ANTHROPIC ============
async function callAnthropic(systemPrompt, messages, options) {
  const anthropicMessages = messages
    .filter(m => m.role !== 'system')
    .map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) }));

  if (anthropicMessages.length === 0) {
    anthropicMessages.push({ role: 'user', content: 'Hello' });
  }

  if (options.stream) {
    return anthropic.messages.stream({
      model: MODEL,
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      system: systemPrompt,
      messages: anthropicMessages
    });
  } else {
    const response = await anthropic.messages.create({
      model: MODEL,
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      system: systemPrompt,
      messages: anthropicMessages
    });
    
    return {
      content: response.content[0]?.text || '',
      usage: {
        prompt_tokens: response.usage?.input_tokens || 0,
        completion_tokens: response.usage?.output_tokens || 0
      }
    };
  }
}

async function* streamAnthropic(stream) {
  for await (const event of stream) {
    if (event.type === 'content_block_delta' && event.delta?.text) {
      yield { content: event.delta.text, done: false };
    } else if (event.type === 'message_stop') {
      yield { content: '', done: true };
    }
  }
}

// ============ GOOGLE ============
async function callGoogle(systemPrompt, messages, options) {
  const model = google.getGenerativeModel({ 
    model: MODEL,
    systemInstruction: systemPrompt
  });

  // Convert to Google format
  const history = [];
  let lastUserMessage = 'Hello';
  
  for (const msg of messages) {
    if (msg.role === 'system') continue;
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    
    if (msg.role === 'user') {
      lastUserMessage = content;
      history.push({ role: 'user', parts: [{ text: content }] });
    } else if (msg.role === 'assistant') {
      history.push({ role: 'model', parts: [{ text: content }] });
    }
  }

  // Remove last user message from history (it becomes the prompt)
  if (history.length > 0 && history[history.length - 1].role === 'user') {
    history.pop();
  }

  const chat = model.startChat({
    history,
    generationConfig: {
      maxOutputTokens: options.max_tokens,
      temperature: options.temperature
    }
  });

  if (options.stream) {
    return chat.sendMessageStream(lastUserMessage);
  } else {
    const result = await chat.sendMessage(lastUserMessage);
    const response = await result.response;
    
    return {
      content: response.text(),
      usage: {
        prompt_tokens: response.usageMetadata?.promptTokenCount || 0,
        completion_tokens: response.usageMetadata?.candidatesTokenCount || 0
      }
    };
  }
}

async function* streamGoogle(stream) {
  for await (const chunk of stream.stream) {
    const text = chunk.text();
    if (text) {
      yield { content: text, done: false };
    }
  }
  yield { content: '', done: true };
}

// ============ OPENAI ============
async function callOpenAI(systemPrompt, messages, options) {
  const openaiMessages = [
    { role: 'system', content: systemPrompt },
    ...messages.filter(m => m.role !== 'system').map(m => ({
      role: m.role,
      content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content)
    }))
  ];

  if (options.stream) {
    return openai.chat.completions.create({
      model: MODEL,
      messages: openaiMessages,
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      stream: true
    });
  } else {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: openaiMessages,
      max_tokens: options.max_tokens,
      temperature: options.temperature
    });
    
    return {
      content: response.choices[0]?.message?.content || '',
      usage: {
        prompt_tokens: response.usage?.prompt_tokens || 0,
        completion_tokens: response.usage?.completion_tokens || 0
      }
    };
  }
}

async function* streamOpenAI(stream) {
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      yield { content, done: false };
    }
    if (chunk.choices[0]?.finish_reason) {
      yield { content: '', done: true };
    }
  }
}

// ============ UNIFIED INTERFACE ============
async function callLLM(systemPrompt, messages, options) {
  switch (PROVIDER) {
    case 'anthropic':
      if (!anthropic) throw new Error('Anthropic not configured');
      return callAnthropic(systemPrompt, messages, options);
    case 'google':
      if (!google) throw new Error('Google not configured');
      return callGoogle(systemPrompt, messages, options);
    case 'openai':
      if (!openai) throw new Error('OpenAI not configured');
      return callOpenAI(systemPrompt, messages, options);
    default:
      throw new Error(`Unknown provider: ${PROVIDER}`);
  }
}

async function* streamLLM(stream) {
  switch (PROVIDER) {
    case 'anthropic': yield* streamAnthropic(stream); break;
    case 'google': yield* streamGoogle(stream); break;
    case 'openai': yield* streamOpenAI(stream); break;
  }
}

// Convert to OpenAI response format
function toOpenAIResponse(result) {
  return {
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: MODEL,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: result.content },
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: result.usage.prompt_tokens,
      completion_tokens: result.usage.completion_tokens,
      total_tokens: result.usage.prompt_tokens + result.usage.completion_tokens
    }
  };
}

function toOpenAIStreamChunk(content, done) {
  return {
    id: `chatcmpl-stream`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: MODEL,
    choices: [{
      index: 0,
      delta: done ? {} : { content },
      finish_reason: done ? 'stop' : null
    }]
  };
}

// Health check
app.get('/health', limiter, (req, res) => {
  res.json({ 
    status: 'ok', 
    provider: PROVIDER,
    model: MODEL,
    auth: API_KEY ? 'enabled' : 'disabled',
    ipWhitelist: ALLOWED_IPS.length > 0 ? 'enabled' : 'disabled'
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', ipWhitelist, authenticate, async (req, res) => {
  const startTime = Date.now();
  
  try {
    const validationError = validateRequest(req.body);
    if (validationError) {
      auditLog(req, 400, { reason: validationError });
      return res.status(400).json({ error: validationError });
    }

    const { messages, stream = false, max_tokens = 1024, temperature = 0.7 } = req.body;
    const memoryContext = buildSystemContext();
    
    // Extract system messages from request
    const systemFromRequest = messages
      .filter(m => m.role === 'system')
      .map(m => m.content)
      .join('\n');
    
    const fullSystemPrompt = memoryContext + '\n' + systemFromRequest;

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const streamResponse = await callLLM(fullSystemPrompt, messages, { stream: true, max_tokens, temperature });
      
      for await (const chunk of streamLLM(streamResponse)) {
        res.write(`data: ${JSON.stringify(toOpenAIStreamChunk(chunk.content, chunk.done))}\n\n`);
      }
      
      res.write('data: [DONE]\n\n');
      res.end();
      
      auditLog(req, 200, { stream: true, durationMs: Date.now() - startTime });
    } else {
      const result = await callLLM(fullSystemPrompt, messages, { stream: false, max_tokens, temperature });
      
      auditLog(req, 200, { 
        stream: false, 
        durationMs: Date.now() - startTime,
        inputTokens: result.usage.prompt_tokens,
        outputTokens: result.usage.completion_tokens
      });

      res.json(toOpenAIResponse(result));
    }
  } catch (error) {
    console.error('Error:', error);
    auditLog(req, 500, { error: error.message });
    res.status(500).json({ error: { message: 'Internal server error', type: 'api_error' } });
  }
});

// Models endpoint
app.get('/v1/models', ipWhitelist, authenticate, (req, res) => {
  res.json({
    object: 'list',
    data: [{
      id: MODEL,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: PROVIDER
    }]
  });
});

app.use((req, res) => res.status(404).json({ error: 'Not found' }));
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Ozzy Voice endpoint running on port ${PORT}`);
  console.log(`Provider: ${PROVIDER}`);
  console.log(`Model: ${MODEL}`);
  console.log(`Memory path: ${MEMORY_PATH}`);
  console.log(`Auth: ${API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`IP whitelist: ${ALLOWED_IPS.length > 0 ? ALLOWED_IPS.join(', ') : 'disabled'}`);
});
