"""
Database module for storing Slack user information
Supports both SQLite (development) and PostgreSQL (production)
"""
import os
import sqlite3
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Check if we should use PostgreSQL (for production)
DATABASE_URL = os.getenv('DATABASE_URL')  # Standard env var used by Heroku/Railway

if DATABASE_URL:
    # PostgreSQL support for production
    try:
        import asyncpg
        USE_POSTGRES = True
        logger.info("Using PostgreSQL database")
    except ImportError:
        logger.warning("asyncpg not installed, falling back to SQLite")
        USE_POSTGRES = False
else:
    USE_POSTGRES = False
    logger.info("Using SQLite database")

class UserDatabase:
    def __init__(self, db_path: str = "slack_users.db"):
        self.db_path = db_path
        self.db_url = DATABASE_URL
        self.pool = None
        
        if USE_POSTGRES:
            # PostgreSQL initialization is async, handled in init_postgres
            pass
        else:
            # SQLite initialization
            self._init_sqlite()
    
    async def initialize(self):
        """Async initialization for PostgreSQL"""
        if USE_POSTGRES and not self.pool:
            await self._init_postgres()
    
    def _init_sqlite(self):
        """Initialize SQLite database and create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS slack_users (
                user_id TEXT PRIMARY KEY,
                real_name TEXT,
                display_name TEXT,
                title TEXT,
                timezone TEXT,
                profile_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create execution details table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_details (
                message_ts TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                execution_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP DEFAULT (datetime(CURRENT_TIMESTAMP, '+7 days'))
            )
        """)
        
        # Create user preferences table for App Home settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                tone_of_voice TEXT DEFAULT 'Professional',
                preferred_city TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        

        
        # Create index for cleanup queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_details_expires 
            ON execution_details(expires_at)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"SQLite database initialized at {self.db_path}")
    
    async def _init_postgres(self):
        """Initialize PostgreSQL database and create tables if they don't exist"""
        # Convert DATABASE_URL to asyncpg format if needed (for Heroku)
        db_url = self.db_url
        if db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        
        self.pool = await asyncpg.create_pool(db_url, min_size=1, max_size=10)
        
        # Create tables
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS slack_users (
                    user_id TEXT PRIMARY KEY,
                    real_name TEXT,
                    display_name TEXT,
                    title TEXT,
                    timezone TEXT,
                    profile_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_details (
                    message_ts TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    execution_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days')
                )
            """)
            
            # Create user preferences table for App Home settings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    tone_of_voice TEXT DEFAULT 'Professional',
                    preferred_city TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            

            
            # Create index for cleanup queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_details_expires 
                ON execution_details(expires_at)
            """)
        
        logger.info("PostgreSQL database initialized")
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information from database"""
        if USE_POSTGRES:
            return await self._get_user_postgres(user_id)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_user_sqlite, user_id)
    
    def _get_user_sqlite(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, real_name, display_name, title, timezone, profile_data, updated_at
            FROM slack_users
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            user_data = dict(row)
            # Parse profile_data JSON if present
            if user_data.get('profile_data'):
                try:
                    user_data['profile_data'] = json.loads(user_data['profile_data'])
                except json.JSONDecodeError:
                    user_data['profile_data'] = {}
            return user_data
        return None
    
    async def _get_user_postgres(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user from PostgreSQL database"""
        await self.initialize()  # Ensure pool is created
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT user_id, real_name, display_name, title, timezone, profile_data, updated_at
                FROM slack_users
                WHERE user_id = $1
            """, user_id)
            
            if row:
                user_data = dict(row)
                # Parse profile_data JSON if present
                if user_data.get('profile_data'):
                    try:
                        user_data['profile_data'] = json.loads(user_data['profile_data'])
                    except json.JSONDecodeError:
                        user_data['profile_data'] = {}
                return user_data
            return None
    
    async def save_user(self, user_id: str, user_info: Dict[str, Any]) -> None:
        """Save or update user information in database"""
        if USE_POSTGRES:
            await self._save_user_postgres(user_id, user_info)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_user_sqlite, user_id, user_info)
    
    def _save_user_sqlite(self, user_id: str, user_info: Dict[str, Any]) -> None:
        """Save user to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract relevant fields from Slack user info
        profile = user_info.get('profile', {})
        real_name = user_info.get('real_name', profile.get('real_name', ''))
        display_name = profile.get('display_name', user_info.get('name', ''))
        title = profile.get('title', '')
        timezone = user_info.get('tz', '')
        
        # Store additional profile data as JSON
        profile_data = json.dumps({
            'team': user_info.get('team_id'),
            'is_admin': user_info.get('is_admin', False),
            'is_owner': user_info.get('is_owner', False),
            'is_primary_owner': user_info.get('is_primary_owner', False),
            'is_bot': user_info.get('is_bot', False),
            'email': profile.get('email', ''),
            'status_text': profile.get('status_text', ''),
            'status_emoji': profile.get('status_emoji', '')
        })
        
        # Insert or update user
        cursor.execute("""
            INSERT INTO slack_users (user_id, real_name, display_name, title, timezone, profile_data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                real_name = excluded.real_name,
                display_name = excluded.display_name,
                title = excluded.title,
                timezone = excluded.timezone,
                profile_data = excluded.profile_data,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, real_name, display_name, title, timezone, profile_data))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved user {user_id} ({real_name}) to SQLite database")
    
    async def _save_user_postgres(self, user_id: str, user_info: Dict[str, Any]) -> None:
        """Save user to PostgreSQL database"""
        await self.initialize()  # Ensure pool is created
        
        # Extract relevant fields from Slack user info
        profile = user_info.get('profile', {})
        real_name = user_info.get('real_name', profile.get('real_name', ''))
        display_name = profile.get('display_name', user_info.get('name', ''))
        title = profile.get('title', '')
        timezone = user_info.get('tz', '')
        
        # Store additional profile data as JSON
        profile_data = json.dumps({
            'team': user_info.get('team_id'),
            'is_admin': user_info.get('is_admin', False),
            'is_owner': user_info.get('is_owner', False),
            'is_primary_owner': user_info.get('is_primary_owner', False),
            'is_bot': user_info.get('is_bot', False),
            'email': profile.get('email', ''),
            'status_text': profile.get('status_text', ''),
            'status_emoji': profile.get('status_emoji', '')
        })
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO slack_users (user_id, real_name, display_name, title, timezone, profile_data, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    real_name = EXCLUDED.real_name,
                    display_name = EXCLUDED.display_name,
                    title = EXCLUDED.title,
                    timezone = EXCLUDED.timezone,
                    profile_data = EXCLUDED.profile_data,
                    updated_at = CURRENT_TIMESTAMP
            """, user_id, real_name, display_name, title, timezone, profile_data)
        
        logger.info(f"Saved user {user_id} ({real_name}) to PostgreSQL database")
    
    async def delete_user(self, user_id: str) -> None:
        """Delete user from database"""
        if USE_POSTGRES:
            await self._delete_user_postgres(user_id)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._delete_user_sqlite, user_id)
    
    def _delete_user_sqlite(self, user_id: str) -> None:
        """Delete user from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM slack_users WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted user {user_id} from SQLite database")
    
    async def _delete_user_postgres(self, user_id: str) -> None:
        """Delete user from PostgreSQL database"""
        await self.initialize()  # Ensure pool is created
        
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM slack_users WHERE user_id = $1", user_id)
        
        logger.info(f"Deleted user {user_id} from PostgreSQL database")
    
    # ========================= EXECUTION DETAILS METHODS =========================
    
    async def save_execution_details(self, message_ts: str, channel_id: str, user_id: str, execution_data: Any) -> None:
        """Save execution details to database with 7-day expiration"""
        execution_json = json.dumps(execution_data)
        
        if USE_POSTGRES:
            await self._save_execution_postgres(message_ts, channel_id, user_id, execution_json)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_execution_sqlite, message_ts, channel_id, user_id, execution_json)
    
    def _save_execution_sqlite(self, message_ts: str, channel_id: str, user_id: str, execution_json: str) -> None:
        """Save execution details to SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO execution_details (message_ts, channel_id, user_id, execution_data)
            VALUES (?, ?, ?, ?)
        """, (message_ts, channel_id, user_id, execution_json))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved execution details for message_ts={message_ts}")
    
    async def _save_execution_postgres(self, message_ts: str, channel_id: str, user_id: str, execution_json: str) -> None:
        """Save execution details to PostgreSQL"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO execution_details (message_ts, channel_id, user_id, execution_data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT(message_ts) DO UPDATE SET
                    execution_data = EXCLUDED.execution_data,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = CURRENT_TIMESTAMP + INTERVAL '7 days'
            """, message_ts, channel_id, user_id, execution_json)
        
        logger.info(f"Saved execution details for message_ts={message_ts}")
    
    async def get_execution_details(self, message_ts: str) -> Optional[Any]:
        """Get execution details from database"""
        if USE_POSTGRES:
            return await self._get_execution_postgres(message_ts)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_execution_sqlite, message_ts)
    
    def _get_execution_sqlite(self, message_ts: str) -> Optional[Any]:
        """Get execution details from SQLite"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Clean up expired entries first
        cursor.execute("DELETE FROM execution_details WHERE expires_at < CURRENT_TIMESTAMP")
        
        cursor.execute("""
            SELECT execution_data FROM execution_details
            WHERE message_ts = ? AND expires_at > CURRENT_TIMESTAMP
        """, (message_ts,))
        
        row = cursor.fetchone()
        conn.commit()
        conn.close()
        
        if row:
            try:
                return json.loads(row['execution_data'])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse execution data for message_ts={message_ts}")
                return None
        return None
    
    async def _get_execution_postgres(self, message_ts: str) -> Optional[Any]:
        """Get execution details from PostgreSQL"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            # Clean up expired entries first
            await conn.execute("DELETE FROM execution_details WHERE expires_at < CURRENT_TIMESTAMP")
            
            row = await conn.fetchrow("""
                SELECT execution_data FROM execution_details
                WHERE message_ts = $1 AND expires_at > CURRENT_TIMESTAMP
            """, message_ts)
            
            if row:
                try:
                    return json.loads(row['execution_data'])
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse execution data for message_ts={message_ts}")
                    return None
            return None
    
    async def cleanup_expired_executions(self) -> int:
        """Clean up expired execution details and return count of deleted records"""
        if USE_POSTGRES:
            return await self._cleanup_executions_postgres()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._cleanup_executions_sqlite)
    
    def _cleanup_executions_sqlite(self) -> int:
        """Clean up expired executions from SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM execution_details WHERE expires_at < CURRENT_TIMESTAMP")
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute("DELETE FROM execution_details WHERE expires_at < CURRENT_TIMESTAMP")
            conn.commit()
            logger.info(f"Cleaned up {count} expired execution details")
        
        conn.close()
        return count
    
    async def _cleanup_executions_postgres(self) -> int:
        """Clean up expired executions from PostgreSQL"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM execution_details WHERE expires_at < CURRENT_TIMESTAMP")
            count = int(result.split()[-1]) if result else 0
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired execution details")
            
            return count

    # ========================= USER PREFERENCES METHODS =========================
    
    async def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences from database"""
        if USE_POSTGRES:
            return await self._get_user_preferences_postgres(user_id)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_user_preferences_sqlite, user_id)
    
    def _get_user_preferences_sqlite(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM user_preferences WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    async def _get_user_preferences_postgres(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user preferences from PostgreSQL database"""
        await self.initialize()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM user_preferences WHERE user_id = $1
            """, user_id)
            
            if row:
                return dict(row)
            return None
    
    async def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Save or update user preferences in database"""
        if USE_POSTGRES:
            await self._save_user_preferences_postgres(user_id, preferences)
        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_user_preferences_sqlite, user_id, preferences)
    
    def _save_user_preferences_sqlite(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Save user preferences to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract preferences
        tone_of_voice = preferences.get('tone_of_voice', 'Professional')
        preferred_city = preferences.get('preferred_city', '')
        
        cursor.execute("""
            INSERT INTO user_preferences (user_id, tone_of_voice, preferred_city, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                tone_of_voice = excluded.tone_of_voice,
                preferred_city = excluded.preferred_city,
                updated_at = CURRENT_TIMESTAMP
        """, (user_id, tone_of_voice, preferred_city))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved preferences for user {user_id}")
    
    async def _save_user_preferences_postgres(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Save user preferences to PostgreSQL database"""
        await self.initialize()
        
        # Extract preferences
        tone_of_voice = preferences.get('tone_of_voice', 'Professional')
        preferred_city = preferences.get('preferred_city', '')
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_preferences (user_id, tone_of_voice, preferred_city, updated_at)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    tone_of_voice = EXCLUDED.tone_of_voice,
                    preferred_city = EXCLUDED.preferred_city,
                    updated_at = CURRENT_TIMESTAMP
            """, user_id, tone_of_voice, preferred_city)
        
        logger.info(f"Saved preferences for user {user_id}")
    

    async def close(self):
        """Close database connections"""
        if USE_POSTGRES and self.pool:
            await self.pool.close()

# Global database instance
user_db = UserDatabase()