import sqlite3

# Connect to SQLite database (or create one)
conn = sqlite3.connect("players.db")
cursor = conn.cursor()

# Create table 'topPlayers'
cursor.execute("""
CREATE TABLE IF NOT EXISTS topPlayers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    hits INTEGER NOT NULL
)
""")

# Insert top 3 players
players_data = [
    ("Pete Rose", 4256),
    ("Ty Cobb", 4189),
    ("Hank Aaron", 3771)
]

cursor.executemany("INSERT INTO topPlayers (name, hits) VALUES (?, ?)", players_data)
conn.commit()
conn.close()

print("✅ Database 'players.db' created and populated successfully.")
