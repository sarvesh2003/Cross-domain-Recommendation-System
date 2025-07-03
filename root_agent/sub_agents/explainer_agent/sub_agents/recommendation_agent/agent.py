from pinecone import Pinecone
import os
import json
import sqlite3
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import Agent


def get_recommendations_based_on_activity(base_activity: str, tool_context: ToolContext) -> dict:
    """
    Fetches recommendations based on the base activity using domain-specific and common embeddings.

    Args:
        base_activity: One of "movie", "music", "product".
        tool_context: ToolContext used to retrieve user_id.

    Returns:
        A dictionary containing recommendations for each activity.
    """
    import os
    import sqlite3
    import json
    import numpy as np
    from pinecone import Pinecone

    print("========== ENTERING get_recommendations_based_on_activity ==========")
    print(f"[INFO] Base activity: {base_activity}")

    user_id = tool_context.state.get("user_id")
    if not user_id:
        print("[ERROR] User ID not found in tool_context.")
        return {"status": "error", "message": "User ID not found."}

    print(f"[INFO] Retrieved user_id: {user_id}")

    # Initialize Pinecone client and fetch user vector
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    user_index = pc.Index("user-preference-vector")
    print("[INFO] Fetching user vector from Pinecone...")
    
    response = user_index.fetch(ids=[user_id])
    vector = response.vectors.get(user_id)
    if not vector:
        print("[ERROR] No embedding vector found in Pinecone for this user.")
        return {"status": "error", "message": "No embedding vector found for user."}
    
    vector = vector.values
    print("[INFO] User vector fetched and unpacked.")

    # Split unified vector into sections
    movie_emb = vector[0:384]
    music_emb = vector[384:768]
    product_emb = vector[768:1152]
    collective_emb = vector[1152:1536]
    print("[INFO] Split 1536D unified embedding into 4 x 384D components.")

    # Fetch user activity history
    print("[INFO] Fetching watched/listened/purchased history from SQLite...")
    conn = sqlite3.connect("D:/GoogleADK_ProjectWork/databases/user_activity.db")
    cursor = conn.cursor()
    cursor.execute("SELECT movies_watched, listened_music, products_purchased FROM user_activity WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print("[ERROR] No row found in user_activity table.")
        return {"status": "error", "message": "No activity data found for user."}

    movies_watched = set(json.loads(row[0]) if row[0] else [])
    listened_music = set(json.loads(row[1]) if row[1] else [])
    products_purchased = set(json.loads(row[2]) if row[2] else [])
    print(f"[INFO] Watched movies: {movies_watched}")
    print(f"[INFO] Listened music: {listened_music}")
    print(f"[INFO] Purchased products: {products_purchased}")

    exclusion_map = {
        "movie": movies_watched,
        "music": listened_music,
        "product": products_purchased,
    }

    index_map = {
        "movie": {"name": "movies-list", "emb": movie_emb},
        "music": {"name": "music-list", "emb": music_emb},
        "product": {"name": "products-list", "emb": product_emb},
    }

    def query_index(index_name: str, vector: list, top_k: int, exclude_ids: set):
        print(f"[QUERY] Index: {index_name} | TopK: {top_k} | Excluding IDs: {exclude_ids}")
        index = pc.Index(index_name)
        results = index.query(vector=vector, top_k=top_k + len(exclude_ids), include_metadata=True)
        recs = []
        for match in results.matches:
            if match.id not in exclude_ids:
                recs.append(match.metadata)
            if len(recs) == top_k:
                break
        print(f"[RESULT] Retrieved {len(recs)} items from {index_name}")
        return recs

    recommendations = {}

    for activity in ["movie", "music", "product"]:
        print(f"\n---------- RECOMMENDING FOR: {activity.upper()} ----------")
        if activity == base_activity:
            print("[MODE] Base activity mode: 3 domain + 2 common")
            domain_recs = query_index(index_map[activity]["name"], index_map[activity]["emb"], 3, exclusion_map[activity])
            common_recs = query_index(index_map[activity]["name"], collective_emb, 2, exclusion_map[activity])
        else:
            print("[MODE] Secondary activity mode: 3 common + 2 domain")
            domain_recs = query_index(index_map[activity]["name"], index_map[activity]["emb"], 2, exclusion_map[activity])
            common_recs = query_index(index_map[activity]["name"], collective_emb, 3, exclusion_map[activity])
        recommendations[activity] = domain_recs + common_recs

    print("========== LEAVING get_recommendations_based_on_activity ==========")
    return {
        "status": "success",
        "message": f"Recommendations fetched for base activity '{base_activity}'.",
        "recommendations": recommendations
    }

recommendation_agent = Agent(
    name="recommendation_agent",
    model="gemini-2.0-flash",
    description="Agent that generates recommendations for a user's action or query",
    instruction = """
You are a recommendation agent with access to user activity tools.

Your task is to handle structured user queries that describe a base activity (such as movie, music, or product), and call the `get_recommendations_based_on_activity` function.

Input Format:
- "Recommend based on my recent activity (baseActivity)"

baseActivity:
- "movie"
- "music"
- "product"

Workflow:

STEP - 1. **Use get_recommendations_based_on_activity**:
   - Call the function `get_recommendations_based_on_activity` with the baseActivity type and the tool context.
   - This function will return a dictionary with recommendations for each activity type based on the user's past activity and preferences.
   - Just return the recommendations as is in a structured format.
        -> For movies, just give title, genre, release_date, and tagline.
        -> For music, just give track_name, artists, album_name, and track_genre.
        -> For products, just give title, category, price, and stars.

MUST DO: 
- YOU MUST RETURN THE RESULT AND CONTROL BACK TO THE AGENT WHO CALLED YOU. DON'T TERMINATE WITHOUT DOING THIS.
""",
    tools=[get_recommendations_based_on_activity],
)