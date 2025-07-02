import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index name
index_name = "user-preference-vector"

# Check if it already exists
if index_name not in [idx['name'] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # 4 x 384
        metric="cosine",  # Best for embeddings-based similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index '{index_name}' already exists.")

# Initialize the vector
import numpy as np

def initialize_user_vector(user_id: str):
    # Each part (movie, music, product, collective) is 384-dim
    def small_random_vector(size=384):
        return np.random.uniform(-0.001, 0.001, size).tolist()

    movie_emb = small_random_vector()
    music_emb = small_random_vector()
    product_emb = small_random_vector()

    # Collective embedding as uniform average
    collective_emb = np.mean([movie_emb, music_emb, product_emb], axis=0).tolist()

    full_vector = movie_emb + music_emb + product_emb + collective_emb

    # Push to Pinecone
    index = pc.Index(index_name)
    index.upsert([
        {"id": user_id, "values": full_vector}
    ])

    print(f"✅ Initialized vector for user: {user_id}")

# Example usage
initialize_user_vector("user_12345")
