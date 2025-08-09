"""
Database module for storing Slack user information
"""
import sqlite3
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class UserDatabase:
    def __init__(self, db_path: str = "slack_users.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create tables if they don't exist"""
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
        logger.info(f"Database initialized at {self.db_path}")
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information from database"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_user_sync, user_id)
    
    def _get_user_sync(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous method to get user from database"""
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
    
    async def save_user(self, user_id: str, user_info: Dict[str, Any]) -> None:
        """Save or update user information in database"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_user_sync, user_id, user_info)
    
    def _save_user_sync(self, user_id: str, user_info: Dict[str, Any]) -> None:
        """Synchronous method to save user to database"""
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
        logger.info(f"Saved user {user_id} ({real_name}) to database")
    
    async def delete_user(self, user_id: str) -> None:
        """Delete user from database"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._delete_user_sync, user_id)
    
    def _delete_user_sync(self, user_id: str) -> None:
        """Synchronous method to delete user from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM slack_users WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted user {user_id} from database")

# Global database instance
user_db = UserDatabase()