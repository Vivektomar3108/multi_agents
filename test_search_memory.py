"""
Test script to verify if search agent remembers conversation history
"""
import asyncio
from app.controllers.search_agent_controller import SearchController
from app.config.mongo import init_db, close_db
from uuid import uuid4

async def test_search_memory():
    """Test if the search agent can remember previous conversations"""
    
    # Initialize database
    print("🔄 Initializing database...")
    await init_db()
    
    # Create controller instance
    print("🔄 Initializing SearchController...")
    controller = SearchController()
    
    # Generate test user and chat IDs (must be valid UUIDs for Weaviate)
    user_id = str(uuid4())
    chat_id = str(uuid4())
    
    print(f"\n📝 Test User ID: {user_id}")
    print(f"📝 Test Chat ID: {chat_id}\n")
    
    try:
        # First query - establish context
        print("=" * 60)
        print("QUERY 1: Asking about Python")
        print("=" * 60)
        query1 = "What is Python programming language?"
        result1 = await controller.run_search(user_id, chat_id, query1)
        print(f"\n✅ Query 1 Response:")
        print(f"Status: {result1.get('status')}")
        if 'data' in result1 and 'response' in result1['data']:
            response_text = result1['data']['response'][:300]  # First 300 chars
            print(f"Response: {response_text}...")
        print("\n")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Second query - test memory recall
        print("=" * 60)
        print("QUERY 2: Testing memory recall")
        print("=" * 60)
        query2 = "What did I just ask you about?"
        result2 = await controller.run_search(user_id, chat_id, query2)
        print(f"\n✅ Query 2 Response:")
        print(f"Status: {result2.get('status')}")
        if 'data' in result2 and 'response' in result2['data']:
            response_text = result2['data']['response']
            print(f"Response: {response_text}")
        print("\n")
        
        # Analyze if memory worked
        print("=" * 60)
        print("MEMORY TEST ANALYSIS")
        print("=" * 60)
        
        if 'data' in result2 and 'response' in result2['data']:
            response = result2['data']['response'].lower()
            
            # Check if the response mentions Python or the previous query
            has_python = 'python' in response
            has_language = 'programming' in response or 'language' in response
            has_context = 'asked' in response or 'previous' in response or 'before' in response
            
            if has_python or (has_language and has_context):
                print("✅ MEMORY WORKING: Agent remembered the previous conversation!")
                print(f"   - Mentions Python: {has_python}")
                print(f"   - Mentions programming/language: {has_language}")
                print(f"   - Shows context awareness: {has_context}")
            else:
                print("❌ MEMORY NOT WORKING: Agent doesn't seem to remember previous query")
        else:
            print("⚠️  Could not analyze - unexpected response format")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print("\n🔄 Closing database connection...")
        await close_db()
        print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_search_memory())
