# API Key Token Tracking & Rate Limiting

Your API now includes comprehensive token tracking and rate limiting features to control Claude token consumption per API key.

## 🔑 API Key Authentication

### Supported Headers
The API accepts API keys in two ways:

1. **Authorization Bearer Token** (Recommended)
```bash
curl -H "Authorization: Bearer your-api-key-here" ...
```

2. **X-API-Key Header**
```bash
curl -H "X-API-Key: your-api-key-here" ...
```

### Anonymous Usage
Requests without API keys are allowed and tracked under a default "anonymous" key with the same limits.

## 📊 Token Limits

### Default Limits (Per API Key)
- **Daily**: 100,000 tokens
- **Monthly**: 2,000,000 tokens
- **Request Limit**: 1,000 requests per day

### Rate Limiting (Global)
- **Tokens per minute**: 20,000
- **Burst size**: 4,000 tokens
- **Queue size**: 50 requests
- **Queue timeout**: 2 minutes

## 🛠 API Endpoints

### Send Message with Token Tracking
```bash
POST /api/send-message
Authorization: Bearer your-api-key

{
  "message": "Your message here"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Agent's response...",
  "metadata": {
    "estimated_tokens": 150,
    "api_key_hash": "abc123...",
    "token_usage": {
      "daily_used": 1500,
      "daily_remaining": 98500,
      "monthly_used": 15000,
      "monthly_remaining": 1985000
    }
  }
}
```

### Check Your Usage
```bash
GET /api/usage
Authorization: Bearer your-api-key
```

**Response:**
```json
{
  "api_key_hash": "abc123...",
  "usage": {
    "daily_tokens_used": 1500,
    "daily_tokens_limit": 100000,
    "daily_tokens_remaining": 98500,
    "monthly_tokens_used": 15000,
    "monthly_tokens_limit": 2000000,
    "monthly_tokens_remaining": 1985000,
    "total_requests": 45,
    "daily_utilization_percent": 1.5,
    "monthly_utilization_percent": 0.75
  }
}
```

## 🔧 Admin Endpoints

### Set Custom Limits for API Key
```bash
POST /api/admin/set-limits/{api_key_hash}
Authorization: Bearer admin-api-key

{
  "daily_limit": 50000,
  "monthly_limit": 1000000
}
```

### Get Usage for Specific API Key
```bash
GET /api/admin/usage/{api_key_hash}
Authorization: Bearer admin-api-key
```

### Get All API Key Usage
```bash
GET /api/admin/all-usage
Authorization: Bearer admin-api-key
```

### Reset API Key Usage
```bash
DELETE /api/admin/reset-usage/{api_key_hash}
Authorization: Bearer admin-api-key
```

## ⚠️ Error Responses

### Token Limit Exceeded
```json
{
  "status": "token_limit_exceeded",
  "error": "Daily token limit exceeded. Remaining: 0 tokens",
  "type": "token_limit_error",
  "estimated_tokens": 150
}
```

### Rate Limit Exceeded
```json
{
  "status": "rate_limited",
  "error": "Token bucket exhausted...",
  "retry_after": "60",
  "message": "API rate limit reached. Please try again shortly.",
  "type": "rate_limit_error"
}
```

## 🧪 Testing

### Test Your Setup
```bash
# Start the API server
python -m uvicorn interfaces.api_interface:app --host 0.0.0.0 --port 8000 --reload

# Run comprehensive tests
python test_api_limits.py
```

### Example Test Requests

**With API Key:**
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Authorization: Bearer test-key-123" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, test my token limits!"}'
```

**Without API Key:**
```bash
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, anonymous usage!"}'
```

**Check Usage:**
```bash
curl -H "Authorization: Bearer test-key-123" \
  "http://localhost:8000/api/usage"
```

## 📈 Monitoring

### Rate Limit Status
```bash
GET /api/rate-limit/status
```

### Agent Status (includes rate limits)
```bash
GET /api/agent/status
```

## 🏗 Implementation Details

### Token Estimation
- Uses rough approximation: `message_length / 4 * 2` (input + output estimate)
- In production, integrate with actual Claude API response token counts

### Storage
- Currently uses in-memory storage (resets on server restart)
- For production, consider Redis or database storage

### Security
- API keys are hashed using SHA-256 for privacy
- Only first 16 characters of hash are stored/displayed
- Admin functions require admin API key (implement proper admin auth for production)

### Reset Schedule
- Daily limits reset at midnight UTC
- Monthly limits reset on the 1st of each month

## 🔄 Customization

### Modify Default Limits
Edit the `default_limits` in `APIKeyTokenTracker.__init__()`:

```python
self.default_limits = {
    'daily_limit': 100000,    # Adjust as needed
    'monthly_limit': 2000000, # Adjust as needed
    'request_limit_daily': 1000
}
```

### Rate Limiting Config
Modify `api_rate_config` in `api_interface.py`:

```python
api_rate_config = RateLimitConfig(
    tokens_per_minute=20000,  # Adjust for your needs
    burst_size=4000,          # Adjust burst capacity
    # ... other settings
)
```

## 📝 Best Practices

1. **Use API Keys**: Always use API keys for production to track usage per client
2. **Monitor Usage**: Regularly check usage statistics to understand consumption patterns  
3. **Set Appropriate Limits**: Adjust limits based on your Claude API plan and usage needs
4. **Handle Errors**: Implement proper retry logic for rate limit and token limit errors
5. **Admin Access**: Secure admin endpoints with proper authentication in production
