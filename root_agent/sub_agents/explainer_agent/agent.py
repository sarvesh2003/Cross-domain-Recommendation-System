from datetime import datetime
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import sqlite3
import json

from .sub_agents.summarizer_agent.agent import summarizer_agent
from .sub_agents.recommendation_agent.agent import recommendation_agent

from pinecone import Pinecone
import os
import litellm

import sqlite3
from google.adk.tools.tool_context import ToolContext

litellm._turn_on_debug()

def increment_step_no(tool_context: ToolContext) -> dict:
    """Increment the current step number in the workflow.

    Args:
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message with the updated step number
    """
    current_step = tool_context.state.get("step_no", 0)
    new_step = current_step + 1
    tool_context.state["step_no"] = new_step

    print(f"--- Tool: increment_step_no called. Step incremented from {current_step} to {new_step} ---")

    return {
        "action": "increment_step_no",
        "step_no": new_step,
        "message": f"Step number incremented to {new_step}",
    }


def calculate_user_embeddings(activity_type: str, user_query: str, description: str, tool_context: ToolContext) -> dict:
    """
    Updates the embedding vector for the user in Pinecone based on the provided activity type.

    Args:
        activity_type: One of "movie", "music", "product".
        user_query: The original user query string.
        description: The generated description of the item.
        tool_context: ToolContext used to retrieve user_id.

    Returns:
        A dictionary indicating success or error.
    """
    print("==================== INSIDE CALCULATE_USER_EMBEDDINGS ====================")
    import os
    import sqlite3
    import json
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
    import numpy as np

    user_id = tool_context.state.get("user_id")
    if not user_id:
        return {"status": "error", "message": "User ID not found."}

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or any other 384-dim model
    new_embedding = model.encode(description).tolist()  # 384-dim

    # DB Connection to fetch list sizes
    conn = sqlite3.connect("D:/GoogleADK_ProjectWork/databases/user_activity.db")
    cursor = conn.cursor()

    cursor.execute("SELECT movies_watched, listened_music, products_purchased FROM user_activity WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"status": "error", "message": f"No user data found for {user_id}."}

    movies_watched = json.loads(row[0]) if row[0] else []
    listened_music = json.loads(row[1]) if row[1] else []
    products_purchased = json.loads(row[2]) if row[2] else []

    # Get counts and compute weights
    counts = {
        "movie": len(movies_watched),
        "music": len(listened_music),
        "product": len(products_purchased),
    }
    total_count = sum(counts.values())
    if total_count == 0:
        return {"status": "error", "message": "No activity history to calculate collective embedding."}

    weights = {k: counts[k] / total_count for k in counts}

    # Pinecone init and fetch
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("user-preference-vector")

    response = index.fetch(ids=[user_id])
    existing_vector = response.vectors.get(user_id)
    if existing_vector:
        existing_vector = existing_vector.values
    else:
        existing_vector = [0.0] * 1536

    # Split current embedding into 4 parts
    movie_emb = existing_vector[0:384]
    music_emb = existing_vector[384:768]
    product_emb = existing_vector[768:1152]
    collective_emb = existing_vector[1152:1536]

    # Update the relevant part
    if activity_type == "movie":
        movie_emb = new_embedding
    elif activity_type == "music":
        music_emb = new_embedding
    elif activity_type == "product":
        product_emb = new_embedding

    # Recalculate collective embedding
    movie_wt, music_wt, product_wt = weights["movie"], weights["music"], weights["product"]
    collective_emb = np.average(
        [movie_emb, music_emb, product_emb],
        axis=0,
        weights=[movie_wt, music_wt, product_wt]
    ).tolist()

    # Concatenate and update full vector
    full_vector = movie_emb + music_emb + product_emb + collective_emb
    index.upsert(vectors=[{"id": user_id, "values": full_vector}])
    print("==================== LEAVING CALCULATE_USER_EMBEDDINGS ====================")

    return {
        "status": "success",
        "message": f"Embedding updated for user {user_id}.",
        "weights": weights
    }


def set_user_pref_summary(activity_type: str, new_summary: str, tool_context: ToolContext) -> dict:
    """
    Updates the user preference summary in the database for the specified activity type.

    Args:
        activity_type: One of "movie", "music", or "product".
        new_summary: The updated preference summary string from the summarizer agent.
        tool_context: ToolContext object used to retrieve user_id.

    Returns:
        A dictionary indicating success or error status.
    """
    print("==================== INSIDE SET_USER_PREF_SUMMARY ====================")

    print("==================== SETTING SUMMARY ====================")
    print(f"Activity Type: {activity_type}")
    print(f"New Summary: {new_summary}")
    print(f"Tool Context: {tool_context}")
    print("==================== END OF INPUT INFO ====================")

    user_id = tool_context.state.get("user_id")
    if not user_id:
        return {
            "action": "set_user_pref_summary",
            "status": "error",
            "message": "User ID not found in tool_context state.",
        }

    activity_column_map = {
        "movie": "movie_pref_summary",
        "music": "music_pref_summary",
        "product": "product_pref_summary",
    }

    column_name = activity_column_map.get(activity_type)
    if not column_name:
        return {
            "action": "set_user_pref_summary",
            "status": "error",
            "message": f"Invalid activity_type: {activity_type}. Must be one of {list(activity_column_map.keys())}.",
        }

    try:
        conn = sqlite3.connect("D:/GoogleADK_ProjectWork/databases/user_activity.db")
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE user_activity SET {column_name} = ? WHERE user_id = ?",
            (new_summary, user_id),
        )
        conn.commit()
        conn.close()
        print("==================== SUMMARY UPDATED SUCCESSFULLY ====================")
        return {
            "action": "set_user_pref_summary",
            "status": "success",
            "message": f"Updated {column_name} for user {user_id}.",
        }
    except Exception as e:
        return {
            "action": "set_user_pref_summary",
            "status": "error",
            "message": f"Database error: {str(e)}",
        }


def get_item_description(activity_type: str, item_name: str, tool_context: ToolContext) -> dict:
    """
    Fetches a structured textual description of an item (movie, music, or product) from Pinecone
    using its title or track name.

    Args:
        activity_type: One of "movie", "music", "product".
        item_name: The name of the movie, track, or product title.
        tool_context: ToolContext with Pinecone API access.

    Returns:
        A dictionary with status and formatted description string.
    """
    from pinecone import Pinecone
    import os
    print("==================== INSIDE GET_ITEM_DESCRIPTION ====================")
    index_map = {
        "movie": ("movies-list", "original_title"),
        "music": ("music-list", "track_name"),
        "product": ("products-list", "title"),
    }

    if activity_type not in index_map:
        return {"status": "error", "message": f"Invalid activity_type: {activity_type}"}

    index_name, filter_field = index_map[activity_type]

    try:
        # Init Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)

        # Dummy vector (not used for scoring, only filtering)
        zero_vector = [0.0] * 384

        response = index.query(
            vector=zero_vector,
            filter={filter_field: {"$eq": item_name}},
            top_k=1,
            include_metadata=True
        )
        matches = response.get("matches", [])
        if not matches:
            return {"status": "not_found", "message": f"No match for {item_name}"}

        metadata = matches[0]["metadata"]

        # Format output string by type
        if activity_type == "movie":
            description = (
                f"Movie: {metadata.get('title')}\n"
                f"Genres: {metadata.get('genres')}\n"
                f"Keywords: {metadata.get('keywords')}\n"
                f"Overview: {metadata.get('overview')}\n"
                f"Tagline: {metadata.get('tagline')}\n"
                f"Release Date: {metadata.get('release_date')}\n"
                f"Popularity Score: {metadata.get('popularity')}\n"
                f"Average Vote: {metadata.get('vote_average')}"
            )
        elif activity_type == "music":
            description = (
                f"Track: {metadata.get('track_name')}\n"
                f"Artist(s): {metadata.get('artists')}\n"
                f"Album: {metadata.get('album_name')}\n"
                f"Genre: {metadata.get('track_genre')}\n"
                f"Explicit: {'Yes' if metadata.get('explicit') else 'No'}\n"
                f"Duration: {round(metadata.get('duration_ms') / 1000)} seconds\n"
                f"Popularity Score: {metadata.get('popularity')}"
            )
        elif activity_type == "product":
            description = (
                f"Product: {metadata.get('title')}\n"
                f"Category: {metadata.get('category')}\n"
                f"Price: ${metadata.get('price')}\n"
                f"List Price: ${metadata.get('listPrice')}\n"
                f"Star Rating: {metadata.get('stars')}⭐\n"
                f"Reviews: {metadata.get('reviews')}\n"
                f"Best Seller: {'Yes' if metadata.get('isBestSeller') else 'No'}\n"
                f"Recently Bought: {metadata.get('boughtInLastMonth')} times"
            )
        else:
            return {"status": "error", "message": "Unexpected activity type."}
        print("==================== FETCHED ITEM DESCRIPTION ====================")
        print(f"Result: {description}")
        print("==================== END OF FETCHED ITEM DESCRIPTION ====================")
        return {
            "status": "success",
            "description": description,
            "item_name": item_name,
            "type": activity_type
        }
        

    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_user_pref_summary(activity_type: str, tool_context: ToolContext) -> dict:
    """
    Fetches current summary for the current user. Given the activity type as movie/music/product, it retrieves the corresponding summary from the database.

    Args:
        activity_type: Type of preference summary to retrieve ("movie", "music", or "product").
        tool_context: ToolContext object used to retrieve user_id.

    Returns:
        A dictionary with the preference summary or error details.
    """
    print("==================== INSIDE GET_USER_PREF_SUMMARY ====================")
    print("==================== FETCHING SUMMARY ====================")
    print(f"Activity Type: {activity_type}")
    print(f"Tool Context: {tool_context}")
    print("==================== END OF INPUT INFO ====================")

    user_id = tool_context.state.get("user_id")
    if not user_id:
        return {
            "action": "get_user_pref_summary",
            "status": "error",
            "message": "User ID not found in tool_context state.",
        }

    activity_column_map = {
        "movie": "movie_pref_summary",
        "music": "music_pref_summary",
        "product": "product_pref_summary",
    }

    column_name = activity_column_map.get(activity_type)
    if not column_name:
        return {
            "action": "get_user_pref_summary",
            "status": "error",
            "message": f"Invalid activity_type: {activity_type}. Must be one of {list(activity_column_map.keys())}.",
        }

    try:
        conn = sqlite3.connect("D:/GoogleADK_ProjectWork/databases/user_activity.db")
        cursor = conn.cursor()
        cursor.execute(f"SELECT {column_name} FROM user_activity WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        print("==================== FETCHED SUMMARY ====================")
        print(f"Result: {result}")
        print("==================== END OF FETCHED SUMMARY ====================")
    except Exception as e:
        return {
            "action": "get_user_pref_summary",
            "status": "error",
            "message": f"Database error: {str(e)}",
        }

    return {
        "action": "get_user_pref_summary",
        "user_id": user_id,
        "type": activity_type,
        "field": column_name,
        "value": result[0] if result else None,
        "status": "success" if result else "not_found",
        "message": f"Fetched {column_name} for user {user_id}." if result else f"No data found for {column_name}.",
    }


def exact_title_search(item: str, field: str):
    
    indexName = ""
    filterBy = ""
    if field == "movies_watched":
        indexName = "movies-list"
        filterBy = "original_title"
    elif field == "products_purchased":
        indexName = "products-list"
        filterBy = "title"
    elif field == "listened_music":
        indexName = "music-list"
        filterBy = "track_name"

    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(indexName)

    try:
        # Using a zero vector since we're not doing vector search
        zero_vector = [0.0] * 384

        response = index.query(
            vector=zero_vector,  # Provide a zero vector of appropriate dimension
            filter={filterBy: {"$eq": item}},  # Explicit equality filter
            top_k=1,
            include_metadata=True
        )

        matches = response.get("matches", [])
        if not matches:
            return {"status": "not_found", "query": item, "message": "No exact match found"}

        return {
            "status": "success",
            "query": item,
            "results": [
                {
                    "id": match["id"],
                    filterBy: match["metadata"].get(filterBy),
                    "genres": match["metadata"].get("genres"),
                    "release_date": match["metadata"].get("release_date"),
                    "tagline": match["metadata"].get("tagline")
                }
                for match in matches
            ]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

# (1)

def update_user_activity(field: str, item: str, tool_context: ToolContext) -> dict:
    """
    Update a list-type activity field (movies_watched, products_purchased, listened_music) for the current user.

    Args:
        field: Field name to update (must be one of the list-type fields)
        item: Item to append to the list
        tool_context: Context for accessing and updating session state (used to get user_id)

    Returns:
        A dictionary with details of the update operation
    """
    print("==================== PRINTING CONTENTS =================")
    print(f"Field: {field}")
    print(f"Item: {item}")
    print(f"Tool Context: {tool_context}")
    print("==================== END OF CONTENTS =================")
    user_id = tool_context.state.get("user_id")

    print(f"--- Tool: update_user_activity called for user_id='{user_id}', field='{field}', item='{item}' ---")

    if not user_id:
        return {
            "action": "update_user_activity",
            "status": "error",
            "message": "User ID not found in tool_context state.",
        }

    # Validate field
    valid_fields = ["movies_watched", "products_purchased", "listened_music"]
    if field not in valid_fields:
        return {
            "action": "update_user_activity",
            "status": "error",
            "message": f"Invalid field: {field}. Must be one of {valid_fields}.",
        }

    results = exact_title_search(item, field)
    print("===================== RESULTS ======================")
    print(results)
    id_from_pinecone = results["results"][0]["id"]
    print(id_from_pinecone)
    print("=====================================================")
    # Connect to the SQLite database
    conn = sqlite3.connect("D:/GoogleADK_ProjectWork/databases/user_activity.db")
    cursor = conn.cursor()

    # Fetch current data
    cursor.execute(f"SELECT {field} FROM user_activity WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return {
            "action": "update_user_activity",
            "status": "error",
            "message": f"No user found with user_id: {user_id}",
        }

    # Parse list from JSON
    current_list = json.loads(row[0]) if row[0] else []

    # Track old value for return info
    old_list = current_list.copy()

    # Append new item if not already present
    if id_from_pinecone not in current_list:
        current_list.append(id_from_pinecone)
        cursor.execute(
            f"UPDATE user_activity SET {field} = ? WHERE user_id = ?",
            (json.dumps(current_list), user_id)
        )
        conn.commit()
        updated = True
    else:
        updated = False  # Already exists, no change made

    conn.close()

    return {
        "action": "update_user_activity",
        "user_id": user_id,
        "field": field,
        "old_value": old_list,
        "new_value": current_list,
        "updated": updated,
        "message": f"{'Added' if updated else 'No change'} to {field} for user {user_id}.",
    }

explainer_agent = Agent(
    name="explainer_agent",
    model="gemini-2.0-flash",
    description="Agent that generates possible explanations for a user's action or query",
    instruction = """
You are an explanation-processing agent with access to user activity tools and a summarization agent.

Your task is to handle structured user queries that describe a recent action (such as watching a movie, purchasing a product, or listening to music), and update relevant user data and summaries accordingly. Your main goal is to proceed with all the flows mentioned in the workflow below, ensuring that you follow each step sequentially without skipping any.
After each step, record the result into a variable, then explicitly check if you have completed all 8 steps before finishing. At each step, print "Currently in step X" to indicate your progress, where X is the step number.

Also, after every step completion, you **must** call the tool `increment_step_no(tool_context)` to update the step count.
At the end of each step, check whether the returned `step_no` equals 9. If yes, stop the workflow immediately, as all 8 steps are now completed.

Input Format:
- Natural language query of the form: "I <action> <item> (source: <Platform>)"

Source Mapping:
- "Amazon Prime" → "movie"
- "Spotify" → "music"
- "Amazon" → "product"

Workflow:

STEP - 1. **Parse and Infer Activity Type**:
   - Extract the item (e.g., "Small Soldiers") and source (e.g., "Amazon Prime") from the input.
   - Infer `activity_type` using the Source Mapping.
   - Ask yourself whether you have any more steps to perform or not.
   - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 2. **Map Activity Type to DB Field**:
   - Map `activity_type` to:
     → "movie" → `movies_watched`
     → "music" → `listened_music`
     → "product" → `products_purchased`
   - Call `update_user_activity(field, item, tool_context)`.
   - Ask yourself whether you have any more steps to perform or not.
   - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 3. **Activity Type to Summary Field Mapping**:
   - Map `activity_type` to summary field:
     → "movie" → `movie_pref_summary`
     → "music" → `music_pref_summary`
     → "product" → `product_pref_summary`
   - Ask yourself whether you have any more steps to perform or not.
   - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 4. **Fetch**:
    - First, fetch the current summary by using the `get_user_pref_summary(activity_type, tool_context)` tool where you pass the activity type as either movie/music/product. 
    - Store the response from the `get_user_pref_summary(activity_type, tool_context)` tool call in `current_summary`.
    - Secondly, fetch the metadata-based description of the item using the tool `get_item_description(activity_type, item_name, tool_context)` where:
        - `activity_type` is one of ["movie", "music", "product"]
        - `item_name` is the exact item string extracted from the user query (e.g., movie title or product name)
    - Store the `description` field from the tool response in a variable called `description_of_query`.

    NOTE: USE THE TOOL `get_user_pref_summary(activity_type, tool_context)` TO GET THE CURRENT SUMMARY AND `get_item_description(activity_type, item_name, tool_context)` TO GET THE DESCRIPTION OF THE USER'S ITEM. ONCE WE GET BOTH THE SUMMARY AND DESCRIPTION, THEN ONLY YOU SHOULD MAKE THE CALL TO SUMMARIZER AGENT - summarizer_agent (YOU HAVE ACCESS TO THIS)
    - Ask yourself whether you have any more steps to perform or not.
    - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 5. **Summarize**:
    - Now, call the summarizer agent (`summarizer_agent`) with the following input tuple:
        → `(current_summary, user_query, description_of_query)`
    IMP: YOU HAVE ACCESS TO THE SUMMARIZER AGENT, SO YOU CAN CALL IT OR TRANSFER THE CONTROL TO IT.
    - Ask yourself whether you have any more steps to perform or not.
    - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 6. **Update Summary in DB**:
    - Use set_user_pref_summary(activity_type, new_summary, tool_context) to update the corresponding summary field (movie_pref_summary, music_pref_summary, or product_pref_summary) in the database.
        - `activity_type` is one of ["movie", "music", "product"]
        - `new_summary` is the summary returned by the summarizer agent.
    NOTE: TO UPDATE THE MOVIE PREFERENCE SUMMARY / MUSIC PREFERENCE SUMMARY / PRODUCT PREFERENCE SUMMARY, USE THE TOOL `set_user_pref_summary(activity_type, new_summary, tool_context)`.
    - Ask yourself whether you have any more steps to perform or not.
    - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 7. **Recalculate Embeddings**:
   - Call `calculate_user_embeddings(activity_type, user_query, description_of_query, tool_context)` to refresh the user vector in Pinecone.
   - The embedding vector has 4 parts (movie_emb, music_emb, product_emb, collective_emb), each 384-dim, stored as one concatenated 1536-dim vector in the "user-preference-vector" index.
   - `collective_emb` is a weighted average of the individual embeddings, with weights proportional to the number of items watched/listened/purchased:
       - movies_watched → `movie_emb`
       - listened_music → `music_emb`
       - products_purchased → `product_emb`
       - Weights = (count / total_count), obtained from the SQLite database.
   - Only the updated section and collective part are recomputed and written back.
   - Ask yourself whether you have any more steps to perform or not.
   - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

STEP - 8. **Call Recommendation Agent**:
   - Once you have updated the embeddings using `calculate_user_embeddings`, immediately trigger a recommendation request to the `recommendation_agent`.
   - Use the same activity_type used throughout the previous steps as the base activity.
   - Format the query as: "Recommend based on my recent activity (baseActivity)"
       - Example: "Recommend based on my recent activity (movie)"
   - Call the `recommendation_agent`, passing this formatted query and you must make sure this step is completed
   - Ask yourself whether you have any more steps to perform or not.
   - ✅ Then call: `increment_step_no(tool_context)` and check if `step_no == 9`. If yes, stop.

Rules:
- Always extract `item` and `source` accurately.
- Do not generate explanations — only structured updates.
- Perform all eight steps sequentially for each query - MUST REQUIREMENT
- After each step, verify whether you have any remaining steps to perform or not. This will ensure that you do not skip any step.

MUST TO FOLLOW INFO NO MATER WHAT: 
- After calling any tool, you MUST use its output.
- Store the value of 'value' from `get_user_pref_summary(...)` into a variable called `current_summary`.
- Store the 'description' field from `get_item_description(...)` into a variable called `description_of_query`.
- Once both variables are populated, call the `summarizer_agent` with: (current_summary, user_query, description_of_query)
- MOST IMPORTANT OF ALL (FAILING TO DO SO WILL COST US MILLIONS): You MUST proceed through all 8 steps, even if some tool returns a success message. DO NOT stop after printing.
""",
#     instruction = """
# You are an explanation-processing agent with access to user activity tools and a summarization agent.

# Your task is to handle structured user queries that describe a recent action (such as watching a movie, purchasing a product, or listening to music), and update relevant user data and summaries accordingly. Your main goal is to proceed with all the flows mentioned in the workflow below, ensuring that you follow each step sequentially without skipping any.
# After each step, record the result into a variable, then explicitly check if you have completed all 8 steps before finishing. At each step, print "Currently in step X" to indicate your progress, where X is the step number.

# Input Format:
# - Natural language query of the form: "I <action> <item> (source: <Platform>)"

# Source Mapping:
# - "Amazon Prime" → "movie"
# - "Spotify" → "music"
# - "Amazon" → "product"

# Workflow:

# STEP - 1. **Parse and Infer Activity Type**:
#    - Extract the item (e.g., "Small Soldiers") and source (e.g., "Amazon Prime") from the input.
#    - Infer `activity_type` using the Source Mapping.
#    - Ask yourself whether you have any more steps to perform or not.

# STEP - 2. **Map Activity Type to DB Field**:
#    - Map `activity_type` to:
#      → "movie" → `movies_watched`
#      → "music" → `listened_music`
#      → "product" → `products_purchased`
#    - Call `update_user_activity(field, item, tool_context)`.
#    - Ask yourself whether you have any more steps to perform or not.

# STEP - 3. **Activity Type to Summary Field Mapping**:
#    - Map `activity_type` to summary field:
#      → "movie" → `movie_pref_summary`
#      → "music" → `music_pref_summary`
#      → "product" → `product_pref_summary`
#    - Ask yourself whether you have any more steps to perform or not.
     
# STEP - 4. **Fetch**:
#     - First, fetch the current summary by using the `get_user_pref_summary(activity_type, tool_context)` tool where you pass the activity type as either movie/music/product. 
#     - Store the response from the `get_user_pref_summary(activity_type, tool_context)` tool call in `current_summary`.
#     - Secondly, fetch the metadata-based description of the item using the tool `get_item_description(activity_type, item_name, tool_context)` where:
#         - `activity_type` is one of ["movie", "music", "product"]
#         - `item_name` is the exact item string extracted from the user query (e.g., movie title or product name)
#     - Store the `description` field from the tool response in a variable called `description_of_query`.
    
#     NOTE: USE THE TOOL `get_user_pref_summary(activity_type, tool_context)` TO GET THE CURRENT SUMMARY AND `get_item_description(activity_type, item_name, tool_context)` TO GET THE DESCRIPTION OF THE USER'S ITEM. ONCE WE GET BOTH THE SUMMARY AND DESCRIPTION, THEN ONLY YOU SHOULD MAKE THE CALL TO SUMMARIZER AGENT - summarizer_agent (YOU HAVE ACCESS TO THIS)
#     - Ask yourself whether you have any more steps to perform or not.
# STEP - 5. **Summarize**:
#     - Now, call the summarizer agent (`summarizer_agent`) with the following input tuple:
#         → `(current_summary, user_query, description_of_query)`
#     IMP: YOU HAVE ACCESS TO THE SUMMARIZER AGENT, SO YOU CAN CALL IT OR TRANSFER THE CONTROL TO IT.
#     - Ask yourself whether you have any more steps to perform or not.

# STEP - 6. **Update Summary in DB**:
#     - Use set_user_pref_summary(activity_type, new_summary, tool_context) to update the corresponding summary field (movie_pref_summary, music_pref_summary, or product_pref_summary) in the database.
#         - `activity_type` is one of ["movie", "music", "product"]
#         - `new_summary` is the summary returned by the summarizer agent.
#     NOTE: TO UPDATE THE MOVIE PREFERENCE SUMMARY / MUSIC PREFERENCE SUMMARY / PRODUCT PREFERENCE SUMMARY, USE THE TOOL `set_user_pref_summary(activity_type, new_summary, tool_context)`.
#         - Ask yourself whether you have any more steps to perform or not.
# STEP - 7. **Recalculate Embeddings**:
#    - Call `calculate_user_embeddings(activity_type, user_query, description_of_query, tool_context)` to refresh the user vector in Pinecone.
#    - The embedding vector has 4 parts (movie_emb, music_emb, product_emb, collective_emb), each 384-dim, stored as one concatenated 1536-dim vector in the "user-preference-vector" index.
#    - `collective_emb` is a weighted average of the individual embeddings, with weights proportional to the number of items watched/listened/purchased:
#        - movies_watched → `movie_emb`
#        - listened_music → `music_emb`
#        - products_purchased → `product_emb`
#        - Weights = (count / total_count), obtained from the SQLite database.
#    - Only the updated section and collective part are recomputed and written back.
#    - Ask yourself whether you have any more steps to perform or not.
# STEP - 8. **Call Recommendation Agent**:
#    - Once you have updated the embeddings using `calculate_user_embeddings`, immediately trigger a recommendation request to the `recommendation_agent`.
#    - Use the same activity_type used throughout the previous steps as the base activity.
#    - Format the query as: "Recommend based on my recent activity (baseActivity)"
#        - Example: "Recommend based on my recent activity (movie)"
#    - Call the `recommendation_agent`, passing this formatted query and you must make sure this step is completed
#    - Ask yourself whether you have any more steps to perform or not.
   
# Rules:
# - Always extract `item` and `source` accurately.
# - Do not generate explanations — only structured updates.
# - Perform all eight steps sequentially for each query - MUST REQUIREMENT
# - After each step, verify whether you have any remaining steps to perform or not. This will ensure that you do not skip any step.


# MUST TO FOLLOW INFO NO MATER WHAT: 
# - After calling any tool, you MUST use its output.
# - Store the value of 'value' from `get_user_pref_summary(...)` into a variable called `current_summary`.
# - Store the 'description' field from `get_item_description(...)` into a variable called `description_of_query`.
# - Once both variables are populated, call the `summarizer_agent` with: (current_summary, user_query, description_of_query)
# - MOST IMPORTANT OF ALL (FAILING TO DO SO WILL COST US MILLIONS): You MUST proceed through all 8 steps, even if some tool returns a success message. DO NOT stop after printing.
# """,
    tools=[update_user_activity, get_user_pref_summary, get_item_description, set_user_pref_summary, calculate_user_embeddings, increment_step_no],
    sub_agents=[summarizer_agent, recommendation_agent],
)