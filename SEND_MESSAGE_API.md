# Send Message API Documentation

## Overview

The `/api/send-message` endpoint provides a comprehensive interface for sending messages to the Signal Agent and receiving AI-powered responses. This endpoint supports conversation context, session management, tool calls, and includes built-in rate limiting, token tracking, and usage monitoring.

**Key Features:**
- 🔄 **Full agent capabilities** - Access all agent features including tool usage
- 💬 **Context support** - Maintain conversation history and session state  
- 🔧 **Tool integration** - See what tools the agent uses to answer your questions
- 📊 **Token tracking** - Real-time usage monitoring per API key
- 🛡️ **Rate limiting** - Built-in protection against overuse
- ⚡ **Graceful errors** - Friendly error responses instead of HTTP exceptions

## 🚀 Quick Start

### Basic Request
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

### With API Key (Recommended)
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Authorization: Bearer sk-signal-your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

## 📋 API Specification

### Endpoint
```
POST /api/send-message
```

### Headers
| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `Authorization` | No | Bearer token with your API key |
| `X-API-Key` | No | Alternative to Authorization header |

### Request Body
```json
{
  "message": "Your message to the AI agent",
  "context": {
    "user_id": "12345",
    "session_id": "abc-123",
    "conversation_history": "Previous conversation context",
    "custom_data": "Any additional context you want to provide"
  },
  "agent_type": "research"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The message you want to send to the AI agent |
| `context` | object | No | Additional context to help the agent understand your request better |
| `agent_type` | string | No | Type of agent to use (default: "research") |

### Response Format

#### Success Response (200)
```json
{
  "status": "success",
  "response": "AI agent's response to your message",
  "tool_calls": [
    {
      "tool": "web_search",
      "parameters": {"query": "..."},
      "result": "..."
    }
  ],
  "context": {
    "updated_session_data": "...",
    "conversation_memory": "...",
    "agent_state": "..."
  },
  "metadata": {
    "request_id": "1234567890",
    "agent_type": "ClientAgent",
    "agent_type_requested": "research",
    "estimated_tokens": 150,
    "api_key_hash": "abc12345def67890",
    "token_usage": {
      "daily_used": 1500,
      "daily_remaining": 98500,
      "monthly_used": 15000,
      "monthly_remaining": 1985000
    }
  }
}
```

#### Error Responses

**Rate Limited (429)**
```json
{
  "status": "rate_limited",
  "error": "Token bucket exhausted...",
  "retry_after": "60",
  "message": "API rate limit reached. Please try again shortly.",
  "estimated_tokens": 150,
  "type": "rate_limit_error"
}
```

**Token Limit Exceeded (200 with error status)**
```json
{
  "status": "token_limit_exceeded",
  "error": "Daily token limit exceeded. Remaining: 0 tokens",
  "type": "token_limit_error",
  "estimated_tokens": 150
}
```

**Invalid API Key (200 with error status)**
```json
{
  "status": "error",
  "error": "Invalid API key",
  "type": "authentication_error"
}
```

**Server Error (500)**
```json
{
  "detail": "Internal server error message"
}
```

## 🔑 Authentication

### Option 1: Authorization Header (Recommended)
```bash
curl -H "Authorization: Bearer sk-signal-your-api-key" ...
```

### Option 2: X-API-Key Header
```bash
curl -H "X-API-Key: sk-signal-your-api-key" ...
```

### Option 3: No Authentication
Requests without an API key are allowed but tracked under an anonymous key with default limits.

## 📊 Rate Limiting & Token Tracking

### Rate Limits (Global)
- **20,000 tokens per minute** across all requests
- **4,000 token burst** capacity
- **50 request queue** with 2-minute timeout

### Token Limits (Per API Key)
- **100,000 tokens per day** per API key
- **2,000,000 tokens per month** per API key
- **1,000 requests per day** per API key

### Token Estimation
- Rough estimate: `message_length / 4 * 2` (input + output)
- Actual usage tracked in response metadata

## 💡 Usage Examples

### Simple Text Message
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-signal-your-key" \
  -d '{
    "message": "What is the capital of France?"
  }'
```

**Response:**
```json
{
  "status": "success",
  "response": "The capital of France is Paris.",
  "tool_calls": [],
  "context": {},
  "metadata": {
    "estimated_tokens": 25,
    "token_usage": {
      "daily_used": 25,
      "daily_remaining": 99975
    }
  }
}
```

### Complex Query with Context
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-signal-your-key" \
  -d '{
    "message": "Can you explain quantum computing and provide a simple example of how quantum algorithms differ from classical ones?",
    "context": {
      "user_id": "user123",
      "session_id": "learning-session-456",
      "previous_topics": ["classical computing", "algorithms"],
      "user_level": "beginner"
    },
    "agent_type": "research"
  }'
```

### Programming Help with Session Context
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-signal-your-key" \
  -d '{
    "message": "Write a Python function that calculates the fibonacci sequence up to n terms",
    "context": {
      "session_id": "coding-session-789",
      "programming_language": "python",
      "skill_level": "intermediate",
      "previous_code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    }
  }'
```

## 🔧 Integration Examples

### Python
```python
import requests
import json

def send_message(message, api_key=None, context=None, agent_type="research"):
    url = "http://localhost:8000/api/send-message"
    headers = {"Content-Type": "application/json"}
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    data = {
        "message": message,
        "context": context or {},
        "agent_type": agent_type
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Simple usage
result = send_message("Hello AI!", "sk-signal-your-key")
print(result["response"])

# With context
context = {
    "user_id": "user123",
    "session_id": "chat-456",
    "conversation_history": "User is learning about AI"
}
result = send_message("What is machine learning?", "sk-signal-your-key", context)
print(result["response"])
print(f"Tool calls: {result['tool_calls']}")
print(f"Updated context: {result['context']}")
```

### JavaScript/Node.js
```javascript
async function sendMessage(message, apiKey = null, context = {}, agentType = "research") {
    const url = 'http://localhost:8000/api/send-message';
    const headers = {
        'Content-Type': 'application/json'
    };
    
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    const payload = {
        message: message,
        context: context,
        agent_type: agentType
    };
    
    const response = await fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}

// Simple usage
sendMessage("Hello AI!", "sk-signal-your-key")
    .then(result => console.log(result.response));

// With context and session management
const context = {
    sessionId: "chat-session-123",
    userId: "user456",
    previousTopics: ["programming", "python"]
};

sendMessage("How do I use async/await?", "sk-signal-your-key", context)
    .then(result => {
        console.log("Response:", result.response);
        console.log("Tool calls:", result.tool_calls);
        console.log("Updated context:", result.context);
    });
```

### PHP
```php
<?php
function sendMessage($message, $apiKey = null, $context = [], $agentType = "research") {
    $url = 'http://localhost:8000/api/send-message';
    $payload = [
        'message' => $message,
        'context' => $context,
        'agent_type' => $agentType
    ];
    $data = json_encode($payload);
    
    $headers = [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($data)
    ];
    
    if ($apiKey) {
        $headers[] = 'Authorization: Bearer ' . $apiKey;
    }
    
    $contextOptions = stream_context_create([
        'http' => [
            'method' => 'POST',
            'header' => implode("\r\n", $headers),
            'content' => $data
        ]
    ]);
    
    $response = file_get_contents($url, false, $contextOptions);
    return json_decode($response, true);
}

// Simple usage
$result = sendMessage("Hello AI!", "sk-signal-your-key");
echo $result['response'];

// With context
$context = [
    'user_id' => 'user123',
    'session_id' => 'php-session-456',
    'user_preferences' => ['detailed_explanations' => true]
];

$result = sendMessage("Explain PHP arrays", "sk-signal-your-key", $context);
echo "Response: " . $result['response'] . "\n";
echo "Tool calls: " . json_encode($result['tool_calls']) . "\n";
echo "Context: " . json_encode($result['context']) . "\n";
?>
```

## 🛡 Security Considerations

### API Key Security
- Store API keys securely (environment variables, secure config)
- Never expose API keys in client-side code
- Rotate keys regularly
- Use different keys for different environments

### Rate Limiting
- Implement retry logic with exponential backoff
- Monitor rate limit headers and queue status
- Handle rate limit errors gracefully

### Input Validation
- Validate message content on your side
- Implement message length limits if needed
- Sanitize user input before sending

## 🔍 Monitoring & Debugging

### Check Your Usage
```bash
curl -H "Authorization: Bearer sk-signal-your-key" \
  "http://localhost:8000/api/usage"
```

### Check Rate Limit Status
```bash
curl "http://localhost:8000/api/rate-limit/status"
```

### Check Server Health
```bash
curl "http://localhost:8000/api/agent/status"
```

## ⚠️ Error Handling Best Practices

### Handle Rate Limits
```python
import time
import requests

def send_with_retry(message, api_key, max_retries=3):
    for attempt in range(max_retries):
        response = send_message(message, api_key)
        
        if response["status"] == "success":
            return response
        elif response["status"] == "rate_limited":
            retry_after = int(response.get("retry_after", 60))
            print(f"Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after)
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
            break
    
    return None
```

### Handle Token Limits
```python
def check_token_usage(api_key):
    response = requests.get(
        "http://localhost:8000/api/usage",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    usage = response.json()["usage"]
    remaining = usage["daily_tokens_remaining"]
    
    if remaining < 1000:  # Less than 1000 tokens left
        print(f"Warning: Only {remaining} tokens remaining today")
        return False
    
    return True
```

## 🔗 Related Endpoints

- **GET `/api/usage`** - Check your token usage
- **GET `/api/agent/status`** - Server health and status
- **GET `/api/rate-limit/status`** - Current rate limiting status
- **WebSocket `/api/chat/stream`** - Real-time streaming chat interface

## 📞 Support

For issues or questions:
1. Check the server logs for error details
2. Verify your API key is valid and active
3. Ensure you're within rate and token limits
4. Check the server status endpoints for system health

---

**Base URL**: `http://localhost:8000` (development)  
**Production URL**: Configure based on your deployment  
**API Version**: 1.0  
**Content-Type**: `application/json`  
**Authentication**: Bearer token (optional but recommended)
