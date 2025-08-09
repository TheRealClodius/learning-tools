"""
Database module for storing Slack user information
Supports both SQLite (development) and PostgreSQL (production)
"""
import os
import sqlite3
import asyncio
import logging
from typing import Optional, Dict, Any
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
        
        # Create table
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
    
    async def close(self):
        """Close database connections"""
        if USE_POSTGRES and self.pool:
            await self.pool.close()

# Global database instance
user_db = UserDatabase()