# Database Deployment Guide

The Slack user caching feature uses a database to store user information (name, role, timezone) to avoid repeated API calls. Here's how the database works in different environments:

## Local Development

- **Database**: SQLite (`slack_users.db` file)
- **Setup**: Automatic - the database file is created on first run
- **Location**: In your project directory
- **No configuration needed**

## Production Deployment

For production, you need a persistent database. The application automatically detects and uses PostgreSQL when the `DATABASE_URL` environment variable is set.

### Supported Platforms with Built-in PostgreSQL

#### 1. **Railway** (Recommended)
```bash
# In Railway dashboard:
1. Add PostgreSQL service to your project
2. Railway automatically sets DATABASE_URL
3. Deploy your app - it will auto-connect
```

#### 2. **Heroku**
```bash
# Add Heroku Postgres addon
heroku addons:create heroku-postgresql:mini

# DATABASE_URL is automatically set
# Deploy as normal
git push heroku main
```

#### 3. **Render**
```bash
# In Render dashboard:
1. Create a PostgreSQL database
2. Copy the "Internal Database URL"
3. Add it as DATABASE_URL env var to your web service
```

#### 4. **Supabase** (Free PostgreSQL)
```bash
# Create free database at supabase.com
# Use the connection string as DATABASE_URL
DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[database]
```

### Manual PostgreSQL Setup

If your platform doesn't provide PostgreSQL:

1. **Use a managed PostgreSQL service**:
   - [Supabase](https://supabase.com) - Free tier available
   - [Neon](https://neon.tech) - Serverless Postgres
   - [ElephantSQL](https://elephantsql.com) - Free tiny turtle plan

2. **Set the DATABASE_URL environment variable**:
   ```bash
   DATABASE_URL=postgresql://username:password@host:5432/dbname
   ```

## How It Works

1. **First Message from User**:
   - System checks if user exists in database
   - If not, fetches info from Slack API
   - Saves to database for future use

2. **Subsequent Messages**:
   - User info loaded from database (fast)
   - No Slack API call needed
   - Includes name, role, and timezone in context

3. **Automatic Fallback**:
   - If PostgreSQL is not available, falls back to SQLite
   - If database operations fail, continues without caching

## Environment Variables

```bash
# For production with PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# All platforms that provide PostgreSQL will set this automatically
```

## Data Stored

For each Slack user:
- User ID (primary key)
- Real name
- Display name  
- Job title/role
- Timezone
- Additional profile data (JSON)
- Created/updated timestamps

## Database Schema

The same schema works for both SQLite and PostgreSQL:

```sql
CREATE TABLE slack_users (
    user_id TEXT PRIMARY KEY,
    real_name TEXT,
    display_name TEXT,
    title TEXT,
    timezone TEXT,
    profile_data TEXT,  -- JSON string
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Troubleshooting

1. **"asyncpg not installed" warning**:
   - This is normal in development
   - The app falls back to SQLite

2. **Database not persisting on Heroku/Railway**:
   - Make sure you've added a PostgreSQL addon
   - Check that DATABASE_URL is set

3. **Connection errors**:
   - Verify DATABASE_URL format
   - Check firewall/security group settings
   - Ensure database is accessible from your app

## Privacy & Security

- User data is cached to improve performance
- Only public Slack profile information is stored
- No messages or private data are saved
- Users can be removed with: `await user_db.delete_user(user_id)`