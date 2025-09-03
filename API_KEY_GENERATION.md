# API Key Generation Guide

Your Signal Agent API now includes a complete API key generation and management system.

## 🔑 Quick Start

### Generate Your First API Key

**Option 1: Command Line (Recommended)**
```bash
# Generate a simple API key
python generate_api_key.py generate --simple

# Generate and store a managed key
python generate_api_key.py generate --name "My Test Key"
```

**Option 2: Using the API (Admin)**
```bash
curl -X POST "http://localhost:8000/api/admin/generate-key" \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Test Key", "prefix": "sk-signal"}'
```

## 📋 Command Line Tool

### Generate API Keys
```bash
# Simple key generation (no storage)
python generate_api_key.py generate --simple

# Full key with storage and management
python generate_api_key.py generate --name "Production API" --prefix "sk-prod"

# Custom length
python generate_api_key.py generate --name "Test Key" --length 48
```

### List All Keys
```bash
python generate_api_key.py list
```

**Example Output:**
```
Found 3 API key(s):
--------------------------------------------------------------------------------
Name: My Test Key
Hash: abc12345def67890
Preview: sk-signal-AbC1...XyZ9
Status: 🟢 Active
Created: 2024-01-15T10:30:00
Usage: 42 requests
Last used: 2024-01-15T15:45:00
--------------------------------------------------------------------------------
```

### Revoke Keys
```bash
# Revoke by hash
python generate_api_key.py revoke abc12345def67890

# Revoke by full key
python generate_api_key.py revoke sk-signal-AbC123XyZ789...
```

### Validate Keys
```bash
python generate_api_key.py validate sk-signal-AbC123XyZ789...
```

## 🌐 API Endpoints for Key Management

### Generate Key (Admin)
```bash
POST /api/admin/generate-key
Authorization: Bearer admin-key

{
  "name": "Client API Key",
  "prefix": "sk-client",
  "length": 32
}
```

**Response:**
```json
{
  "status": "success",
  "message": "API key generated successfully",
  "api_key": "sk-client-AbC123XyZ789...",
  "name": "Client API Key",
  "hash": "abc12345def67890",
  "created": "2024-01-15T10:30:00",
  "warning": "Store this key securely - it won't be shown again!"
}
```

### List All Keys (Admin)
```bash
GET /api/admin/list-keys
Authorization: Bearer admin-key
```

### Revoke Key (Admin)
```bash
DELETE /api/admin/revoke-key/abc12345def67890
Authorization: Bearer admin-key
```

## 🔒 API Key Security Features

### Key Format
- **Prefix**: `sk-signal-` (customizable)
- **Random Part**: 32 characters by default (letters + numbers)
- **Example**: `sk-signal-AbC123XyZ789DeF456GhI789JkL012`

### Security Features
- **Cryptographically Secure**: Uses `secrets` module for generation
- **Hashed Storage**: Keys are hashed (SHA-256) for internal tracking
- **Revocation**: Keys can be instantly revoked
- **Usage Tracking**: Monitor key usage and last access
- **Validation**: Optional validation of keys before processing requests

## 📊 Integration with Token Tracking

Generated API keys automatically integrate with the token tracking system:

```bash
# Use your generated key
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Authorization: Bearer sk-signal-AbC123XyZ789..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello with my generated key!"}'
```

**Response includes usage:**
```json
{
  "status": "success",
  "response": "Hello! How can I help you?",
  "metadata": {
    "api_key_hash": "abc12345def67890",
    "estimated_tokens": 25,
    "token_usage": {
      "daily_used": 1250,
      "daily_remaining": 98750,
      "monthly_used": 15000,
      "monthly_remaining": 1985000
    }
  }
}
```

## 🛠 API Key Storage

### File Storage
Keys are stored in `api_keys.json`:
```json
{
  "keys": {
    "sk-signal-AbC123...": {
      "name": "My Test Key",
      "created": "2024-01-15T10:30:00",
      "hash": "abc12345def67890",
      "active": true,
      "usage_count": 42,
      "last_used": "2024-01-15T15:45:00"
    }
  },
  "metadata": {
    "created": "2024-01-15T10:00:00",
    "last_generated": "2024-01-15T10:30:00"
  }
}
```

### Production Considerations
- **Database Storage**: For production, consider using a database instead of JSON files
- **Encryption**: Encrypt the keys file if storing sensitive information
- **Backup**: Regular backups of key data
- **Rotation**: Implement key rotation policies

## 🎯 Usage Examples

### For Testing Platforms
```bash
# Generate a key for your testing platform
python generate_api_key.py generate --name "Testing Platform Key"

# Use the generated key
curl -X POST "http://localhost:8000/api/send-message" \
  -H "Authorization: Bearer sk-signal-[YOUR-KEY]" \
  -H "Content-Type: application/json" \
  -d '{"message": "Test message from platform"}'
```

### For Different Clients
```bash
# Generate keys for different clients/projects
python generate_api_key.py generate --name "Web Frontend" --prefix "sk-web"
python generate_api_key.py generate --name "Mobile App" --prefix "sk-mobile"
python generate_api_key.py generate --name "Data Pipeline" --prefix "sk-data"
```

### Monitor Usage
```bash
# Check usage for specific key
curl -H "Authorization: Bearer sk-signal-[YOUR-KEY]" \
  "http://localhost:8000/api/usage"

# Admin: View all key usage
curl -H "Authorization: Bearer admin-key" \
  "http://localhost:8000/api/admin/all-usage"
```

## 🔧 Configuration

### Customize Default Settings
Edit `generate_api_key.py`:
```python
# Change default prefix
def generate_simple_key(prefix: str = "sk-mycompany", length: int = 32):

# Change default length
gen_parser.add_argument("--length", type=int, default=48, help="Length of random part")
```

### Enable/Disable Validation
In `api_interface.py`, the system automatically detects if the key manager is available:
- **With validation**: Keys are checked against the stored database
- **Without validation**: Any string is accepted as an API key

## 🚨 Error Handling

### Invalid API Key
```json
{
  "error": "Invalid API key",
  "message": "The provided API key is invalid or has been revoked",
  "type": "authentication_error"
}
```

### Key Generation Errors
```json
{
  "status": "error",
  "detail": "API key management not available. Import generate_api_key module."
}
```

## 📚 Best Practices

1. **Use Descriptive Names**: Always name your keys descriptively
2. **Regular Rotation**: Rotate keys periodically for security
3. **Monitor Usage**: Check key usage regularly for anomalies
4. **Revoke Unused Keys**: Remove keys that are no longer needed
5. **Secure Storage**: Store keys securely on the client side
6. **Environment Variables**: Use environment variables for production keys
7. **Separate Keys**: Use different keys for different environments (dev/staging/prod)

## 🔄 Migration from Manual Keys

If you were using manual/simple API keys before:
1. Generate proper keys using the tool
2. Update your clients to use the new keys
3. Enable validation by ensuring `generate_api_key.py` is available
4. Revoke old manual keys once migration is complete
