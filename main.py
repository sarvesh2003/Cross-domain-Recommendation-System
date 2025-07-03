import asyncio

# Import the root agent
######################################### from customer_service_agent.agent import customer_service_agent
from root_agent.agent import root_agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from helper import add_user_query_to_history, call_agent_async
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ===== PART 1: Initialize Persistent Session Service =====
# Using SQLite database for persistent storage
db_url = "sqlite:///./databases/agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)


# ===== PART 2: Initialize Pinecone DB =====
def init_pinecone_client():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment")
    
    pc = Pinecone(api_key=api_key)
    
    return {
        "movies": pc.Index("movies-list"),
        "music": pc.Index("music-list"),
        "products": pc.Index("products-list")
    }

# ===== PART 3: Define Initial State =====
# This will only be used when creating a new session
initial_state = {
    "user_name": "",
    "user_id": "",
    "step_no": 0,
    "pinecone_indices": ["movies-list", "music-list", "products-list"],
}


async def main_async(userID):
    # Setup constants
    APP_NAME = "Inter-domain Recommendation Engine"
    USER_ID = userID

    initial_state["user_id"] = USER_ID
    initial_state["step_no"] = 0
    
    # ===== PART 4: Session Management - Find or Create =====
    # Check for existing sessions for this user
    existing_sessions = session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID,
    )

    # If there's an existing session, use it, otherwise create a new one
    if existing_sessions and len(existing_sessions.sessions) > 0:
        # Use the most recent session
        SESSION_ID = existing_sessions.sessions[0].id
        print(f"Continuing existing session: {SESSION_ID}")
    else:
        # Create a new session with initial state
        new_session = session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state,
        )
        SESSION_ID = new_session.id
        print(f"Created new session: {SESSION_ID}")
    
    # ===== PART 3B: Initialize Pinecone Indices =====
    pinecone_indices = init_pinecone_client()
    # ===== PART 4: Agent Runner Setup =====
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        # shared_state={"pinecone_indices": pinecone_indices},
    )
    # ===== PART 5: Interactive Conversation Loop =====
    print("\nWelcome to Recommendation Engine Agent Chat!")
    print("Your preferences will be remembered across conversations.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    # Set the user variable if not already set
    user_input = "Do you know my name, if not ask me to introduce myself"
    await call_agent_async(runner, USER_ID, SESSION_ID, user_input)

    while True:
        # Get user input
        user_input = input("You: ")
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation. Your data has been saved to the database.")
            break
        # Process the user query through the agent
        await call_agent_async(runner, USER_ID, SESSION_ID, user_input)

if __name__ == "__main__":
    user_id_input = input("Enter your user ID: ")
    asyncio.run(main_async(userID=user_id_input))
