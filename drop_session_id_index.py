"""
Migration script to drop old session_id index from chat_sessions collection.

The schema now uses 'chat_id' instead of 'session_id', but MongoDB still has 
the old unique index which causes duplicate key errors.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.setting import settings


async def drop_old_index():
    """Drop the session_id_1 index from chat_sessions collection."""
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client[settings.mongo_db_name]
    collection = db["chat_sessions"]
    
    try:
        # List current indexes
        print("Current indexes:")
        indexes = await collection.list_indexes().to_list(length=None)
        for idx in indexes:
            print(f"  - {idx['name']}: {idx.get('key', {})}")
        
        # Drop the old session_id_1 index if it exists
        print("\nDropping session_id_1 index...")
        await collection.drop_index("session_id_1")
        print("✓ Successfully dropped session_id_1 index")
        
        # Show remaining indexes
        print("\nRemaining indexes:")
        indexes = await collection.list_indexes().to_list(length=None)
        for idx in indexes:
            print(f"  - {idx['name']}: {idx.get('key', {})}")
            
    except Exception as e:
        if "index not found" in str(e).lower():
            print("✓ Index session_id_1 does not exist (already removed)")
        else:
            print(f"✗ Error: {e}")
            raise
    finally:
        client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Drop old session_id index")
    print("=" * 60)
    asyncio.run(drop_old_index())
    print("\n✓ Migration complete!")
