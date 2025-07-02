import sqlite3
import json

# Unique user ID
USER_ID = "user_12345"

# Connect to SQLite database (or create it)
conn = sqlite3.connect("../databases/user_activity.db")
cursor = conn.cursor()

# Create the table with user_id as primary key
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_activity (
    user_id TEXT PRIMARY KEY,
    movies_watched TEXT,
    products_purchased TEXT,
    listened_music TEXT,
    music_pref_summary TEXT,
    movie_pref_summary TEXT,
    product_pref_summary TEXT
)
""")

# Sample data
movies_watched = [8587, 10191, 278927]
products_purchased = ["B07WPRQMZH", "B074WP9WVQ", "B0006LT3XS"]
listened_music = ["4gDajIG4yNgBtym4zgtfRe", "2m2HiaiQeTVrLjC4g2KzOz", "1NGjMdwfetEiNK3qn8TyGD"]

music_summary = "listens to anime music a lot"
movie_summary = "watches animated movies"
product_summary = "purchased comfortable underwears"

# Insert or replace the data for the user_id
cursor.execute("""
INSERT OR REPLACE INTO user_activity (
    user_id,
    movies_watched,
    products_purchased,
    listened_music,
    music_pref_summary,
    movie_pref_summary,
    product_pref_summary
) VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    USER_ID,
    json.dumps(movies_watched),
    json.dumps(products_purchased),
    json.dumps(listened_music),
    music_summary,
    movie_summary,
    product_summary
))

# Commit and close
conn.commit()
conn.close()
