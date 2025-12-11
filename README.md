## Agentic Cross-Domain Recommendation System
An intelligent, conversational recommendation engine powered by Google ADK, Gemini 2.0 Flash, and Pinecone. This system utilizes a hierarchical multi-agent architecture to understand user activities (watching movies, listening to music, buying products) and generate cross-domain recommendations by maintaining a dynamic, multi-modal user preference vector.

### Key Features
- **Hierarchical Agent Architecture:** A root_agent coordinates specialized sub-agents for explaining logic (explainer_agent), summarization (summarizer_agent), and retrieval (recommendation_agent).
- **Cross-Domain Vector Logic:** Uses a 1536-dimensional composite vector (combining Movie, Music, Product, and Collective embeddings) to find semantically related items across different categories.
- **Dynamic User Profiling:**
    - Textual Memory: Maintains structured summaries (e.g., "Loved Movies," "Did not like Products") in SQLite.
    - Vector Memory: Updates Pinecone embeddings in real-time based on user sentiment.
- **8-Step Logic Pipeline:** Enforces a strict workflow from parsing user intent to recalculating embeddings and fetching recommendations.

### Tech Stack
- **Orchestration:** Google ADK (Agent Development Kit)
- **LLM:** Gemini 2.0 Flash
- **Vector Database:** Pinecone
- **Persistent Storage:** SQLite
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384d)
- **Language:** Python 3.10+

### Setup & Installation
1. Prerequisites
    - Python 3.10+ installed in your machine.
    - A Pinecone API Key
    - A Google GenAI API Key (for Gemini).
2. Install Dependencies
    ```text
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. Environment Configuration
- Create a .env file in the root directory and edit those values (provided a sample)
4. Database & Vector Initialization

    (NOTE: USE THIS STEP ONLY WHEN YOU WANT TO USE THIS APPLICATION AS AN ADMIN. FOR REGULAR USERS WHO WANT TO USE THIS SYSTEM, YOU DON'T HAVE TO DO THIS STEP)
- Step A: Initialize SQLite Creates the local database to store user history and text summaries.
    ```text
    python user_activity_db_builder.py
    ```
- Step B: Creates the user-preference-vector (1536d) in Pinecone.
    ```text
    python user_pref_index_generation.py
    ```
- Step C: Ingest Item Data Run the following Jupyter notebooks to vectorize and upload the catalog data to Pinecone:
    - movie-recommendation-index-generation.ipynb -> Index: movies-list
    - music-recommendation-index-generation.ipynb -> Index: music-list
    - product-recommendation-index-generation.ipynb -> Index: products-list

### An Example Interaction
```text
> Enter your user ID: user_12345

Welcome to Recommendation Engine Agent Chat!

You: I just watched Interstellar and I absolutely loved the space exploration aspect.

[System]: 
1. Updates 'movies_watched' in SQLite.
2. Summarizer adds "Loves Sci-Fi/Space" to profile.
3. User Vector updated in Pinecone.
4. Recommendation Agent queries Music and Products.

Agent Response: 
"Since you loved 'Interstellar', here are some recommendations:
- Music: 'Cornfield Chase' by Hans Zimmer (Soundtrack)
- Product: Celestron Telescope (Space exploration gear)
- Movie: The Martian"
```

### Sequence Diagram for an example interaction