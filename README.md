# Agentic Cross-Domain Recommendation Engine

A multi-agent recommendation system built with Google ADK and Gemini 2.0 Flash that generates personalized suggestions across Movies, Music, and Products by maintaining a unified user preference vector.

## Problem Statement

Traditional recommendation systems treat each domain independently, so movie preferences don’t affect product recommendations. This project enables **cross-domain recommendations** using a shared embedding space, allowing user interests to transfer across domains (e.g., recommending a telescope to someone loved *Interstellar*).


## Architecture

```
User Query → Root Agent → Explainer Agent (8-step workflow)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Summarizer Agent    Recommendation Agent
                    ↓                   ↓
              SQLite (Text)      Pinecone (Vectors)
```

**Core Components:**
- **Root Agent**: Session management and query routing
- **Explainer Agent**: 8-step stateful workflow orchestrating the pipeline
- **Summarizer Agent**: Converts natural language feedback → structured preference profiles
- **Recommendation Agent**: Vector similarity search across 30K items

## Key Technical Decisions

| Challenge | Solution |
|-----------|----------|
| Cross-domain similarity | 1536-dim composite embedding (4×384): `[movie_vec, music_vec, product_vec, collective_vec]` |
| Preference evolution | Weighted collective vector updated after each interaction |
| Redundancy prevention | Metadata-based exclusion filters in Pinecone |
| Sentiment tracking | Structured summaries with Loved/Okish/Disliked categorization |

## System Flow

![alt text](https://github.com/sarvesh2003/Cross-domain-Recommendation-System/blob/main/sequence_diagram.png)

## Example Interaction

```
You: I just watched Interstellar and loved the space exploration aspect.

Agent: Based on your interest in space exploration sci-fi, here are my recommendations:

Movies:
- The Martian (Sci-Fi, 2015) - "Bring Him Home"
- Gravity (Sci-Fi, 2013) - "Don't Let Go"

Music:
- "Cornfield Chase" by Hans Zimmer
- "Starman" by David Bowie

Products:
- Celestron PowerSeeker Telescope - $79.99 ⭐4.5
- NASA Space Exploration Book - $24.99 ⭐4.8
```


**8-Step Processing Chain:**
1. Parse user input & infer activity type
2. Update domain-specific activity list (SQLite)
3. Fetch item metadata from Pinecone
4. Retrieve current preference summary
5. Generate updated summary via Summarizer Agent
6. Persist new summary to SQLite
7. Recalculate weighted user embedding
8. Query Recommendation Agent for top-k suggestions

## Tech Stack

| Layer | Technology |
|-------|------------|
| Agent Orchestration | Google ADK |
| LLM | Gemini 2.0 Flash |
| Vector Database | Pinecone (4 indices) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Persistence | SQLite + JSON serialization |
| Language | Python 3.10+ |

## Project Structure

```
├── main.py                     # Entry point & session management
├── helper.py                   # Utilities & state management
├── root_agent/
│   └── agent.py               # Query routing & user identification
├── sub_agents/
│   ├── explainer_agent/       # 8-step workflow orchestration
│   ├── summarizer_agent/      # NL → structured preference conversion
│   └── recommendation_agent/  # Vector retrieval & ranking
├── databases/
│   └── user_activity.db       # SQLite persistence
└── notebooks/
    ├── movie-recommendation-index-generation.ipynb
    ├── music-recommendation-index-generation.ipynb
    └── product-recommendation-index-generation.ipynb
```

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/yourusername/cross-domain-rec-engine.git
cd cross-domain-rec-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your PINECONE_API_KEY and GOOGLE_API_KEY

# 3. Initialize databases (first-time setup)
python user_activity_db_builder.py
python user_pref_index_generation.py

# 4. Run
python main.py
```

## Future Improvements

- [ ] Add evaluation metrics (precision@k, diversity)
- [ ] Implement A/B testing framework
- [ ] Add conversation memory beyond single session
- [ ] Batch embedding updates for efficiency
- [ ] Deploy as API with FastAPI




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
