#!/usr/bin/env python3
"""
Quick script to check if the database is working and show its contents
"""
import os
import asyncio
import asyncpg
from datetime import datetime

async def check_database():
    # Get the DATABASE_URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        print("   Make sure you're running this in Railway or have set the DATABASE_URL")
        return
    
    print(f"✅ DATABASE_URL found")
    print(f"   URL prefix: {database_url[:30]}...")
    
    # Convert postgres:// to postgresql:// if needed
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    try:
        # Connect to database
        print("\n🔄 Connecting to PostgreSQL...")
        conn = await asyncpg.connect(database_url)
        print("✅ Connected successfully!")
        
        # Check if slack_users table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'slack_users'
            );
        """)
        
        if not table_exists:
            print("\n❌ Table 'slack_users' does not exist")
            print("   The table will be created when the app starts")
        else:
            print("\n✅ Table 'slack_users' exists")
            
            # Count users
            user_count = await conn.fetchval("SELECT COUNT(*) FROM slack_users")
            print(f"\n📊 Total users in database: {user_count}")
            
            if user_count > 0:
                # Show some users
                print("\n👥 Recent users:")
                rows = await conn.fetch("""
                    SELECT user_id, real_name, display_name, title, timezone, updated_at 
                    FROM slack_users 
                    ORDER BY updated_at DESC 
                    LIMIT 5
                """)
                
                for row in rows:
                    print(f"\n   User ID: {row['user_id']}")
                    print(f"   Name: {row['real_name']} ({row['display_name']})")
                    print(f"   Title: {row['title'] or 'N/A'}")
                    print(f"   Timezone: {row['timezone']}")
                    print(f"   Last updated: {row['updated_at']}")
            else:
                print("\n   No users stored yet. Users are added when they interact with the Slack bot.")
        
        await conn.close()
        print("\n✅ Database check complete!")
        
    except Exception as e:
        print(f"\n❌ Error connecting to database: {e}")
        print("\n   This might mean:")
        print("   - The DATABASE_URL is incorrect")
        print("   - The database service is not running")
        print("   - There's a network issue")

if __name__ == "__main__":
    asyncio.run(check_database())