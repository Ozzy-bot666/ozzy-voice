import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { readFileSync, existsSync, appendFileSync } from 'fs';
import { join } from 'path';

const app = express();

// Security middleware
app.use(helmet());
app.use(express.json({ limit: '100kb' })); // Limit payload size

// Config
const PORT = process.env.PORT || 3000;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const MODEL = process.env.MODEL || 'claude-sonnet-4-20250514';
const MEMORY_PATH = process.env.MEMORY_PATH || '/home/node/clawd';
const API_KEY = process.env.API_KEY;
const ALLOWED_IPS = process.env.ALLOWED_IPS?.split(',').map(ip => ip.trim()) || [];
const LOG_REQUESTS = process.env.LOG_REQUESTS === 'true';

// Initialize Anthropic client
const anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });

// Rate limiting - 60 requests per minute per IP
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 60,
  message: { error: 'Too many requests, please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req) => {
    // Use X-Forwarded-For for proxied requests (Render)
    return req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip;
  }
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
    userAgent: req.headers['user-agent'],
    status,
    ...details
  };
  
  console.log('[AUDIT]', JSON.stringify(entry));
  
  // Optionally write to file
  try {
    const logPath = join(MEMORY_PATH, 'logs', 'voice-audit.log');
    appendFileSync(logPath, JSON.stringify(entry) + '\n');
  } catch (e) {
    // Ignore file write errors
  }
}

// IP whitelist middleware
function ipWhitelist(req, res, next) {
  if (ALLOWED_IPS.length === 0) return next(); // No whitelist = allow all
  
  const clientIp = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip;
  
  // Check if IP matches any allowed pattern
  const allowed = ALLOWED_IPS.some(allowedIp => {
    if (allowedIp === '*') return true;
    if (allowedIp.endsWith('*')) {
      // Prefix match (e.g., "52.0.*" matches "52.0.1.2")
      const prefix = allowedIp.slice(0, -1);
      return clientIp.startsWith(prefix);
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
  if (!API_KEY) {
    auditLog(req, 200, { auth: 'disabled' });
    return next();
  }
  
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    auditLog(req, 401, { reason: 'Missing auth header' });
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }
  
  const token = authHeader.substring(7);
  if (token !== API_KEY) {
    auditLog(req, 401, { reason: 'Invalid API key' });
    return res.status(401).json({ error: 'Invalid API key' });
  }
  
  auditLog(req, 200, { auth: 'success' });
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
        content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
      });
    }
  }

  return { systemPrompt, messages };
}

// Validate request body
function validateRequest(body) {
  if (!body || typeof body !== 'object') {
    return 'Invalid request body';
  }
  
  if (!body.messages || !Array.isArray(body.messages)) {
    return 'messages array is required';
  }
  
  if (body.messages.length > 100) {
    return 'Too many messages (max 100)';
  }
  
  for (const msg of body.messages) {
    if (!msg.role || !['system', 'user', 'assistant'].includes(msg.role)) {
      return 'Invalid message role';
    }
    if (msg.content === undefined || msg.content === null) {
      return 'Message content is required';
    }
  }
  
  if (body.max_tokens && (body.max_tokens < 1 || body.max_tokens > 8192)) {
    return 'max_tokens must be between 1 and 8192';
  }
  
  if (body.temperature && (body.temperature < 0 || body.temperature > 2)) {
    return 'temperature must be between 0 and 2';
  }
  
  return null;
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

// Health check (no auth required, but rate limited)
app.get('/health', limiter, (req, res) => {
  res.json({ 
    status: 'ok', 
    model: MODEL,
    auth: API_KEY ? 'enabled' : 'disabled',
    ipWhitelist: ALLOWED_IPS.length > 0 ? 'enabled' : 'disabled'
  });
});

// OpenAI-compatible chat completions endpoint
app.post('/v1/chat/completions', ipWhitelist, authenticate, async (req, res) => {
  const startTime = Date.now();
  
  try {
    // Validate request
    const validationError = validateRequest(req.body);
    if (validationError) {
      auditLog(req, 400, { reason: validationError });
      return res.status(400).json({ error: validationError });
    }

    const { messages, stream = false, max_tokens = 1024, temperature = 0.7 } = req.body;

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
      
      auditLog(req, 200, { stream: true, durationMs: Date.now() - startTime });
    } else {
      // Non-streaming response
      const response = await anthropic.messages.create({
        model: MODEL,
        max_tokens: max_tokens,
        temperature: temperature,
        system: fullSystemPrompt,
        messages: anthropicMessages
      });

      auditLog(req, 200, { 
        stream: false, 
        durationMs: Date.now() - startTime,
        inputTokens: response.usage?.input_tokens,
        outputTokens: response.usage?.output_tokens
      });

      res.json(toOpenAIResponse(response, MODEL));
    }
  } catch (error) {
    console.error('Error:', error);
    auditLog(req, 500, { error: error.message });
    res.status(500).json({ 
      error: { 
        message: 'Internal server error', 
        type: 'api_error' 
      } 
    });
  }
});

// Models endpoint (for compatibility)
app.get('/v1/models', ipWhitelist, authenticate, (req, res) => {
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

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Ozzy Voice endpoint running on port ${PORT}`);
  console.log(`Model: ${MODEL}`);
  console.log(`Memory path: ${MEMORY_PATH}`);
  console.log(`Auth: ${API_KEY ? 'enabled' : 'disabled'}`);
  console.log(`IP whitelist: ${ALLOWED_IPS.length > 0 ? ALLOWED_IPS.join(', ') : 'disabled'}`);
  console.log(`Request logging: ${LOG_REQUESTS ? 'enabled' : 'disabled'}`);
});
