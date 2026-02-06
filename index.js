import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import OpenAI from 'openai';
import helmet from 'helmet';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const app = express();
app.use(helmet());
app.use(express.json({ limit: '100kb' }));

// Config
const PORT = process.env.PORT || 3000;
const PROVIDER = process.env.PROVIDER || 'anthropic';
const MODEL = process.env.MODEL || getDefaultModel(PROVIDER);
const MEMORY_PATH = process.env.MEMORY_PATH || '/home/node/clawd';
const LOG_REQUESTS = process.env.LOG_REQUESTS === 'true';
const MEMORY_SECRET = process.env.MEMORY_SECRET || ''; // For /memory endpoint auth
const TOOLS_ENABLED = process.env.TOOLS_ENABLED !== 'false'; // Enable by default

// Security metrics (read-only from external perspective)
const securityMetrics = {
  startTime: Date.now(),
  requests: {
    total: 0,
    authenticated: 0,
    failed_auth: 0,
    by_endpoint: {},
    by_ip: {}  // Track requests per IP
  },
  lastFailedAuth: null,
  recentFailedAuths: [], // Last 100, for pattern detection
  suspiciousUserAgents: [], // Detected suspicious user agents
  wsConnections: {
    total: 0,
    active: 0
  }
};

// Suspicious user agent patterns
const SUSPICIOUS_UA_PATTERNS = [
  /sqlmap/i, /nikto/i, /nmap/i, /masscan/i, /zgrab/i,
  /scanner/i, /exploit/i, /injection/i, /attack/i,
  /^python-requests/i, /^curl\//i, /^wget\//i, /libwww-perl/i,
  /havij/i, /acunetix/i, /nessus/i, /burp/i, /dirbuster/i
];

function isSuspiciousUserAgent(ua) {
  if (!ua) return false;
  return SUSPICIOUS_UA_PATTERNS.some(pattern => pattern.test(ua));
}

// Track request for security metrics
function trackRequest(endpoint, authenticated, ip, userAgent) {
  securityMetrics.requests.total++;
  securityMetrics.requests.by_endpoint[endpoint] = (securityMetrics.requests.by_endpoint[endpoint] || 0) + 1;
  
  // Track by IP
  const normalizedIp = ip ? ip.replace(/^::ffff:/, '') : 'unknown';
  securityMetrics.requests.by_ip[normalizedIp] = (securityMetrics.requests.by_ip[normalizedIp] || 0) + 1;
  
  if (authenticated) {
    securityMetrics.requests.authenticated++;
  }
  
  // Check for suspicious user agent
  if (userAgent && isSuspiciousUserAgent(userAgent)) {
    securityMetrics.suspiciousUserAgents.push({
      time: Date.now(),
      ip: normalizedIp,
      userAgent: userAgent.substring(0, 200), // Truncate
      endpoint
    });
    // Keep last 50
    if (securityMetrics.suspiciousUserAgents.length > 50) {
      securityMetrics.suspiciousUserAgents.shift();
    }
  }
}

function trackFailedAuth(endpoint, ip, userAgent) {
  securityMetrics.requests.failed_auth++;
  securityMetrics.lastFailedAuth = new Date().toISOString();
  
  const normalizedIp = ip ? ip.replace(/^::ffff:/, '') : 'unknown';
  
  // Keep last 100 failed auths for pattern analysis
  securityMetrics.recentFailedAuths.push({
    time: Date.now(),
    endpoint,
    ip: normalizedIp,
    userAgent: userAgent ? userAgent.substring(0, 100) : null
  });
  if (securityMetrics.recentFailedAuths.length > 100) {
    securityMetrics.recentFailedAuths.shift();
  }
}

// Auth check helper with tracking
function checkAuth(req, res, endpoint) {
  const authHeader = req.headers.authorization;
  const userAgent = req.headers['user-agent'];
  
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      trackFailedAuth(endpoint, req.ip, userAgent);
      trackRequest(endpoint, false, req.ip, userAgent);
      res.status(401).json({ error: 'Unauthorized' });
      return false;
    }
  }
  trackRequest(endpoint, true, req.ip, userAgent);
  return true;
}

// Cached memory (updated via API)
let cachedMemory = {
  soul: '',
  user: '',
  memory: '',
  today: '',
  updatedAt: null
};

// Pending actions (to be picked up by Clawdbot)
let pendingActions = [];

// Tool definitions for Gemini
const toolDefinitions = [
  {
    name: 'create_reminder',
    description: 'Set a reminder for the user. They will receive a phone call at the specified time. Use for "remind me" requests.',
    parameters: {
      type: 'object',
      properties: {
        message: {
          type: 'string',
          description: 'What to remind about (e.g., "call mom", "take medication")'
        },
        time: {
          type: 'string', 
          description: 'When to remind. Use natural language like "in 20 minutes", "at 5pm", "tomorrow at 9am", or ISO format.'
        }
      },
      required: ['message', 'time']
    }
  },
  {
    name: 'send_message',
    description: 'Send a text message to the user via Telegram. Use when they ask to "send", "text", or "message" something.',
    parameters: {
      type: 'object',
      properties: {
        message: {
          type: 'string',
          description: 'The message content to send'
        }
      },
      required: ['message']
    }
  },
  {
    name: 'add_todo',
    description: 'Add a task to the todo list. Use when user says "add todo", "add task", "remind me to do", or similar.',
    parameters: {
      type: 'object',
      properties: {
        task: {
          type: 'string',
          description: 'The task description'
        },
        needs_research: {
          type: 'boolean',
          description: 'Set to true if this task requires research (e.g., "research best CRM", "find out about X"). Default false.'
        }
      },
      required: ['task']
    }
  },
  {
    name: 'list_todos',
    description: 'List current todo items. Use when user asks "what\'s on my todo list", "my tasks", "what do I need to do".',
    parameters: {
      type: 'object',
      properties: {
        status: {
          type: 'string',
          description: 'Filter by status: "all", "pending", "researching", "done". Default "pending".'
        }
      },
      required: []
    }
  },
  {
    name: 'complete_todo',
    description: 'Mark a todo item as complete. Use when user says "mark X as done", "completed X", "finished X".',
    parameters: {
      type: 'object',
      properties: {
        task_query: {
          type: 'string',
          description: 'Part of the task name to identify which todo to complete'
        }
      },
      required: ['task_query']
    }
  }
];

// Convert to Gemini format
const geminiTools = [{
  functionDeclarations: toolDefinitions.map(t => ({
    name: t.name,
    description: t.description,
    parameters: t.parameters
  }))
}];

// Cached todos (synced via API)
let cachedTodos = { todos: [], lastUpdated: null };

// Execute a tool and return result
async function executeTool(name, params, callId) {
  const action = {
    id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    tool: name,
    params,
    callId,
    createdAt: new Date().toISOString(),
    status: 'pending'
  };
  
  log(`[${callId}] Tool call: ${name}`, params);
  
  switch (name) {
    case 'create_reminder':
      pendingActions.push(action);
      return { 
        success: true, 
        message: `Reminder scheduled for ${params.time}`,
        action_id: action.id
      };
      
    case 'send_message':
      pendingActions.push(action);
      return { 
        success: true, 
        message: 'Message queued for delivery',
        action_id: action.id
      };
    
    case 'add_todo': {
      const todo = {
        id: `todo-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
        task: params.task,
        type: params.needs_research ? 'research' : 'task',
        status: params.needs_research ? 'pending_research' : 'pending',
        created: new Date().toISOString(),
        research: null,
        approved: false
      };
      cachedTodos.todos.push(todo);
      cachedTodos.lastUpdated = new Date().toISOString();
      
      // Queue action for Clawdbot to persist and potentially start research
      action.todo = todo;
      pendingActions.push(action);
      
      const researchNote = params.needs_research ? ' Research team will investigate.' : '';
      return { 
        success: true, 
        message: `Added to todo list: "${params.task}".${researchNote}`,
        todo_id: todo.id
      };
    }
    
    case 'list_todos': {
      const status = params.status || 'pending';
      let todos = cachedTodos.todos;
      
      if (status !== 'all') {
        todos = todos.filter(t => t.status.includes(status) || t.status === status);
      }
      
      if (todos.length === 0) {
        return { 
          success: true, 
          message: 'Your todo list is empty.',
          todos: []
        };
      }
      
      const summaries = todos.map((t, i) => `${i + 1}. ${t.task}${t.type === 'research' ? ' (research)' : ''}`);
      return { 
        success: true, 
        message: `You have ${todos.length} item${todos.length > 1 ? 's' : ''}: ${summaries.join('; ')}`,
        todos: todos
      };
    }
    
    case 'complete_todo': {
      const query = params.task_query.toLowerCase();
      const todo = cachedTodos.todos.find(t => 
        t.task.toLowerCase().includes(query) && t.status !== 'done'
      );
      
      if (!todo) {
        return { 
          success: false, 
          message: `Could not find a pending todo matching "${params.task_query}"`
        };
      }
      
      todo.status = 'done';
      todo.completed = new Date().toISOString();
      cachedTodos.lastUpdated = new Date().toISOString();
      
      // Queue action for Clawdbot to persist
      action.todo_id = todo.id;
      pendingActions.push(action);
      
      return { 
        success: true, 
        message: `Marked as done: "${todo.task}"`,
        todo_id: todo.id
      };
    }
      
    default:
      return { success: false, error: `Unknown tool: ${name}` };
  }
}

// Provider API keys
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

// Initialize LLM clients
let anthropic, google, openai;
if (ANTHROPIC_API_KEY) anthropic = new Anthropic({ apiKey: ANTHROPIC_API_KEY });
if (GOOGLE_API_KEY) google = new GoogleGenerativeAI(GOOGLE_API_KEY);
if (OPENAI_API_KEY) openai = new OpenAI({ apiKey: OPENAI_API_KEY });

function log(...args) {
  if (LOG_REQUESTS) console.log(new Date().toISOString(), ...args);
}

// Build system context from memory files or cache
function buildSystemContext() {
  let context = `You are Ozzy, a snarky AI bat assistant in a voice conversation.
Keep responses concise and conversational - this is spoken, not written.
Be natural, use contractions, don't be overly formal.
Respond in the same language as the user speaks to you.

`;

  // Use cached memory if available (pushed via API)
  if (cachedMemory.updatedAt) {
    if (cachedMemory.soul) context += `\n## SOUL.md\n${cachedMemory.soul}\n`;
    if (cachedMemory.user) context += `\n## USER.md\n${cachedMemory.user}\n`;
    if (cachedMemory.memory) context += `\n## MEMORY.md\n${cachedMemory.memory}\n`;
    if (cachedMemory.today) context += `\n## Today\n${cachedMemory.today}\n`;
    log('Using cached memory from', cachedMemory.updatedAt);
    return context;
  }

  // Fallback: try to read from filesystem (for local dev)
  const files = ['SOUL.md', 'USER.md', 'MEMORY.md'];
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
      context += `\n## Today (${today})\n${readFileSync(todayPath, 'utf-8')}\n`;
    } catch (e) {}
  }

  return context;
}

// Convert Retell transcript to messages format
function transcriptToMessages(transcript) {
  return transcript.map(t => ({
    role: t.role === 'agent' ? 'assistant' : 'user',
    content: t.content
  }));
}

// ============ LLM PROVIDERS ============

async function callLLM(systemPrompt, messages) {
  switch (PROVIDER) {
    case 'anthropic': return callAnthropic(systemPrompt, messages);
    case 'google': return callGoogle(systemPrompt, messages);
    case 'openai': return callOpenAI(systemPrompt, messages);
    default: throw new Error(`Unknown provider: ${PROVIDER}`);
  }
}

async function* streamLLM(systemPrompt, messages, callId = 'unknown') {
  switch (PROVIDER) {
    case 'anthropic': yield* streamAnthropic(systemPrompt, messages); break;
    case 'google': yield* streamGoogle(systemPrompt, messages, callId); break;
    case 'openai': yield* streamOpenAI(systemPrompt, messages); break;
  }
}

async function callAnthropic(systemPrompt, messages) {
  if (!anthropic) throw new Error('Anthropic not configured');
  const response = await anthropic.messages.create({
    model: MODEL,
    max_tokens: 200,
    temperature: 0.7,
    system: systemPrompt,
    messages: messages.length > 0 ? messages : [{ role: 'user', content: 'Hello' }]
  });
  return response.content[0]?.text || '';
}

async function* streamAnthropic(systemPrompt, messages) {
  if (!anthropic) throw new Error('Anthropic not configured');
  const stream = await anthropic.messages.stream({
    model: MODEL,
    max_tokens: 200,
    temperature: 0.7,
    system: systemPrompt,
    messages: messages.length > 0 ? messages : [{ role: 'user', content: 'Hello' }]
  });
  for await (const event of stream) {
    if (event.type === 'content_block_delta' && event.delta?.text) {
      yield event.delta.text;
    }
  }
}

async function callGoogle(systemPrompt, messages) {
  if (!google) throw new Error('Google not configured');
  const model = google.getGenerativeModel({ model: MODEL, systemInstruction: systemPrompt });
  
  const history = [];
  let lastMessage = 'Hello';
  
  for (const msg of messages) {
    if (msg.role === 'user') {
      lastMessage = msg.content;
      history.push({ role: 'user', parts: [{ text: msg.content }] });
    } else {
      history.push({ role: 'model', parts: [{ text: msg.content }] });
    }
  }
  
  if (history.length > 0 && history[history.length - 1].role === 'user') {
    history.pop();
  }
  
  const chat = model.startChat({ history, generationConfig: { maxOutputTokens: 200, temperature: 0.7 } });
  const result = await chat.sendMessage(lastMessage);
  return result.response.text();
}

async function* streamGoogle(systemPrompt, messages, callId = 'unknown') {
  if (!google) throw new Error('Google not configured');
  
  const modelConfig = { 
    model: MODEL, 
    systemInstruction: systemPrompt
  };
  
  // Add tools if enabled
  if (TOOLS_ENABLED) {
    modelConfig.tools = geminiTools;
  }
  
  const model = google.getGenerativeModel(modelConfig);
  
  const history = [];
  let lastMessage = 'Hello';
  
  for (const msg of messages) {
    if (msg.role === 'user') {
      lastMessage = msg.content;
      history.push({ role: 'user', parts: [{ text: msg.content }] });
    } else {
      history.push({ role: 'model', parts: [{ text: msg.content }] });
    }
  }
  
  if (history.length > 0 && history[history.length - 1].role === 'user') {
    history.pop();
  }
  
  const chat = model.startChat({ 
    history, 
    generationConfig: { maxOutputTokens: 200, temperature: 0.7 }
  });
  
  // First call - might be text or function call
  const result = await chat.sendMessage(lastMessage);
  const response = result.response;
  
  // Check for function calls
  const functionCalls = response.functionCalls();
  
  if (functionCalls && functionCalls.length > 0) {
    // Execute tool(s) and get final response
    for (const fc of functionCalls) {
      log(`[${callId}] Function call detected: ${fc.name}`);
      const toolResult = await executeTool(fc.name, fc.args, callId);
      
      // Send function result back to model
      const followUp = await chat.sendMessage([{
        functionResponse: {
          name: fc.name,
          response: toolResult
        }
      }]);
      
      // Stream the final response
      const finalText = followUp.response.text();
      if (finalText) {
        // Yield in chunks for smoother streaming
        const words = finalText.split(' ');
        for (const word of words) {
          yield word + ' ';
        }
      }
    }
  } else {
    // No function call, stream the text response
    const text = response.text();
    if (text) {
      const words = text.split(' ');
      for (const word of words) {
        yield word + ' ';
      }
    }
  }
}

async function callOpenAI(systemPrompt, messages) {
  if (!openai) throw new Error('OpenAI not configured');
  const response = await openai.chat.completions.create({
    model: MODEL,
    max_tokens: 200,
    temperature: 0.7,
    messages: [
      { role: 'system', content: systemPrompt },
      ...(messages.length > 0 ? messages : [{ role: 'user', content: 'Hello' }])
    ]
  });
  return response.choices[0]?.message?.content || '';
}

async function* streamOpenAI(systemPrompt, messages) {
  if (!openai) throw new Error('OpenAI not configured');
  const stream = await openai.chat.completions.create({
    model: MODEL,
    max_tokens: 200,
    temperature: 0.7,
    stream: true,
    messages: [
      { role: 'system', content: systemPrompt },
      ...(messages.length > 0 ? messages : [{ role: 'user', content: 'Hello' }])
    ]
  });
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) yield content;
  }
}

// ============ RETELL WEBSOCKET HANDLER ============

function handleRetellConnection(ws, callId) {
  log(`[${callId}] WebSocket connected`);
  
  const systemPrompt = buildSystemContext();
  
  // Send initial config
  ws.send(JSON.stringify({
    response_type: 'config',
    config: {
      auto_reconnect: true,
      call_details: false
    }
  }));
  
  // Send initial greeting (empty = wait for user to speak first)
  ws.send(JSON.stringify({
    response_type: 'response',
    response_id: 0,
    content: '',
    content_complete: true
  }));
  
  ws.on('message', async (data) => {
    try {
      const event = JSON.parse(data.toString());
      log(`[${callId}] Received:`, event.interaction_type);
      
      switch (event.interaction_type) {
        case 'ping_pong':
          // Respond to ping
          ws.send(JSON.stringify({
            response_type: 'ping_pong',
            timestamp: Date.now()
          }));
          break;
          
        case 'update_only':
          // Just an update, no response needed
          break;
          
        case 'response_required':
        case 'reminder_required':
          // Generate response
          await handleResponseRequired(ws, callId, event, systemPrompt);
          break;
          
        default:
          log(`[${callId}] Unknown interaction_type:`, event.interaction_type);
      }
    } catch (e) {
      console.error(`[${callId}] Error processing message:`, e);
    }
  });
  
  ws.on('close', () => {
    log(`[${callId}] WebSocket closed`);
  });
  
  ws.on('error', (err) => {
    console.error(`[${callId}] WebSocket error:`, err);
  });
}

async function handleResponseRequired(ws, callId, event, systemPrompt) {
  const { response_id, transcript } = event;
  
  try {
    const messages = transcriptToMessages(transcript || []);
    log(`[${callId}] Generating response for response_id=${response_id}, messages=${messages.length}`);
    
    // Stream the response
    let fullContent = '';
    for await (const chunk of streamLLM(systemPrompt, messages, callId)) {
      fullContent += chunk;
      
      // Send partial response
      ws.send(JSON.stringify({
        response_type: 'response',
        response_id,
        content: chunk,
        content_complete: false
      }));
    }
    
    // Send final response marker
    ws.send(JSON.stringify({
      response_type: 'response',
      response_id,
      content: '',
      content_complete: true
    }));
    
    log(`[${callId}] Response complete: ${fullContent.substring(0, 50)}...`);
    
  } catch (e) {
    console.error(`[${callId}] LLM error:`, e);
    
    // Send error response
    ws.send(JSON.stringify({
      response_type: 'response',
      response_id,
      content: "Sorry, I'm having trouble responding right now.",
      content_complete: true
    }));
  }
}

// ============ HTTP ENDPOINTS ============

// CORS middleware for web call endpoint
const webCallCors = (req, res, next) => {
  const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:5173',
    'https://ozzy-frontend.onrender.com'
  ];
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  next();
};

// Create web call - proxy to Retell API
app.options('/create-web-call', webCallCors);
app.post('/create-web-call', webCallCors, async (req, res) => {
  trackRequest('/create-web-call', false, req.ip, req.headers['user-agent']);
  
  const { agent_id, metadata, retell_llm_dynamic_variables } = req.body;
  
  if (!process.env.RETELL_API_KEY) {
    console.error('RETELL_API_KEY not configured');
    return res.status(500).json({ error: 'Server configuration error' });
  }
  
  const payload = { agent_id };
  if (metadata) payload.metadata = metadata;
  if (retell_llm_dynamic_variables) payload.retell_llm_dynamic_variables = retell_llm_dynamic_variables;
  
  try {
    const response = await fetch('https://api.retellai.com/v2/create-web-call', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.RETELL_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      console.error('Retell API error:', data);
      return res.status(response.status).json(data);
    }
    
    res.status(201).json(data);
  } catch (error) {
    console.error('Error creating web call:', error.message);
    res.status(500).json({ error: 'Failed to create web call', details: error.message });
  }
});

app.get('/health', (req, res) => {
  trackRequest('/health', false, req.ip);
  
  // Calculate failed auths in last hour
  const oneHourAgo = Date.now() - 3600000;
  const failedAuthsLastHour = securityMetrics.recentFailedAuths.filter(a => a.time > oneHourAgo).length;
  
  res.json({
    status: 'ok',
    provider: PROVIDER,
    model: MODEL,
    websocket: 'enabled',
    uptime: Math.floor((Date.now() - securityMetrics.startTime) / 1000),
    tools: {
      enabled: TOOLS_ENABLED,
      available: toolDefinitions.map(t => t.name)
    },
    memory: {
      cached: !!cachedMemory.updatedAt,
      updatedAt: cachedMemory.updatedAt
    },
    todos: {
      count: cachedTodos.todos.length,
      pending: cachedTodos.todos.filter(t => t.status.includes('pending')).length,
      lastUpdated: cachedTodos.lastUpdated
    },
    pendingActions: pendingActions.filter(a => a.status === 'pending').length,
    security: {
      requests: securityMetrics.requests.total,
      failedAuths: securityMetrics.requests.failed_auth,
      failedAuthsLastHour,
      wsConnections: securityMetrics.wsConnections.active
    }
  });
});

// Memory sync endpoint - POST to update cached memory
app.post('/memory', (req, res) => {
  if (!checkAuth(req, res, '/memory')) return;
  
  const { soul, user, memory, today } = req.body;
  
  cachedMemory = {
    soul: soul || cachedMemory.soul,
    user: user || cachedMemory.user,
    memory: memory || cachedMemory.memory,
    today: today || cachedMemory.today,
    updatedAt: new Date().toISOString()
  };
  
  log('Memory updated:', {
    soul: cachedMemory.soul?.length || 0,
    user: cachedMemory.user?.length || 0,
    memory: cachedMemory.memory?.length || 0,
    today: cachedMemory.today?.length || 0
  });
  
  res.json({ 
    ok: true, 
    updatedAt: cachedMemory.updatedAt,
    sizes: {
      soul: cachedMemory.soul?.length || 0,
      user: cachedMemory.user?.length || 0,
      memory: cachedMemory.memory?.length || 0,
      today: cachedMemory.today?.length || 0
    }
  });
});

// Memory debug endpoint - GET to check current state
app.get('/memory', (req, res) => {
  // Auth check
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  res.json({
    updatedAt: cachedMemory.updatedAt,
    sizes: {
      soul: cachedMemory.soul?.length || 0,
      user: cachedMemory.user?.length || 0,
      memory: cachedMemory.memory?.length || 0,
      today: cachedMemory.today?.length || 0
    }
  });
});

// ============ SECURITY ENDPOINT ============

// GET detailed security metrics (auth required)
app.get('/security', (req, res) => {
  if (!checkAuth(req, res, '/security')) return;
  
  const oneHourAgo = Date.now() - 3600000;
  const recentFailedAuths = securityMetrics.recentFailedAuths.filter(a => a.time > oneHourAgo);
  
  // Analyze IP patterns
  const ipCounts = {};
  for (const auth of recentFailedAuths) {
    ipCounts[auth.ip] = (ipCounts[auth.ip] || 0) + 1;
  }
  
  // Find suspicious IPs (multiple failed auths)
  const suspiciousIps = Object.entries(ipCounts)
    .filter(([ip, count]) => count >= 3)
    .map(([ip, count]) => ({ ip, failedAttempts: count }))
    .sort((a, b) => b.failedAttempts - a.failedAttempts);
  
  // Recent suspicious user agents
  const recentSuspiciousUAs = securityMetrics.suspiciousUserAgents
    .filter(ua => ua.time > oneHourAgo);
  
  res.json({
    timestamp: new Date().toISOString(),
    uptime: Math.floor((Date.now() - securityMetrics.startTime) / 1000),
    summary: {
      totalRequests: securityMetrics.requests.total,
      authenticatedRequests: securityMetrics.requests.authenticated,
      totalFailedAuths: securityMetrics.requests.failed_auth,
      failedAuthsLastHour: recentFailedAuths.length,
      suspiciousIpsLastHour: suspiciousIps.length,
      suspiciousUAsLastHour: recentSuspiciousUAs.length,
      activeWsConnections: securityMetrics.wsConnections.active
    },
    alerts: {
      bruteForceDetected: recentFailedAuths.length > 10,
      suspiciousIpsDetected: suspiciousIps.length > 0,
      suspiciousUAsDetected: recentSuspiciousUAs.length > 0
    },
    details: {
      suspiciousIps,
      recentFailedAuths: recentFailedAuths.slice(-20), // Last 20
      suspiciousUserAgents: recentSuspiciousUAs.slice(-10),
      requestsByEndpoint: securityMetrics.requests.by_endpoint,
      topIps: Object.entries(securityMetrics.requests.by_ip)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([ip, count]) => ({ ip, requests: count }))
    }
  });
});

// ============ PENDING ACTIONS ENDPOINTS ============

// GET pending actions (for Clawdbot to poll)
app.get('/actions', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  // Return pending actions and clear them
  const actions = pendingActions.filter(a => a.status === 'pending');
  res.json({ actions, count: actions.length });
});

// POST mark action as processed
app.post('/actions/:id/complete', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  const { id } = req.params;
  const action = pendingActions.find(a => a.id === id);
  
  if (action) {
    action.status = 'completed';
    action.completedAt = new Date().toISOString();
    res.json({ ok: true, action });
  } else {
    res.status(404).json({ error: 'Action not found' });
  }
});

// DELETE clear all completed actions (cleanup)
app.delete('/actions/completed', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  const before = pendingActions.length;
  pendingActions = pendingActions.filter(a => a.status === 'pending');
  res.json({ ok: true, removed: before - pendingActions.length });
});

// ============ TODO ENDPOINTS ============

// GET all todos
app.get('/todos', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  res.json(cachedTodos);
});

// POST sync todos from Clawdbot
app.post('/todos', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  const { todos } = req.body;
  if (todos) {
    cachedTodos.todos = todos;
    cachedTodos.lastUpdated = new Date().toISOString();
    log('Todos synced:', todos.length, 'items');
  }
  
  res.json({ ok: true, count: cachedTodos.todos.length, lastUpdated: cachedTodos.lastUpdated });
});

// PATCH update a specific todo (e.g., add research results)
app.patch('/todos/:id', (req, res) => {
  const authHeader = req.headers.authorization;
  if (MEMORY_SECRET) {
    if (!authHeader || authHeader !== `Bearer ${MEMORY_SECRET}`) {
      return res.status(401).json({ error: 'Unauthorized' });
    }
  }
  
  const { id } = req.params;
  const updates = req.body;
  
  const todo = cachedTodos.todos.find(t => t.id === id);
  if (!todo) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  
  Object.assign(todo, updates);
  cachedTodos.lastUpdated = new Date().toISOString();
  
  res.json({ ok: true, todo });
});

// Catch-all for unmatched routes
app.use((req, res) => {
  res.status(404).json({ error: 'Not found. WebSocket endpoint: wss://host/llm-websocket/:call_id' });
});

// ============ SERVER SETUP ============

const server = createServer(app);

const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (request, socket, head) => {
  const pathname = request.url?.split('?')[0] || '';
  console.log('WebSocket upgrade request:', pathname);
  
  if (pathname.startsWith('/llm-websocket/')) {
    wss.handleUpgrade(request, socket, head, (ws) => {
      wss.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

wss.on('connection', (ws, req) => {
  // Extract call_id from path
  const pathname = req.url?.split('?')[0] || '';
  const match = pathname.match(/\/llm-websocket\/([^\/]+)/);
  const callId = match ? match[1] : 'unknown';
  handleRetellConnection(ws, callId);
});

server.listen(PORT, () => {
  console.log(`Ozzy Voice (Retell WebSocket) running on port ${PORT}`);
  console.log(`Provider: ${PROVIDER}`);
  console.log(`Model: ${MODEL}`);
  console.log(`WebSocket endpoint: wss://host/llm-websocket/:call_id`);
  console.log(`Health check: http://host/health`);
});
