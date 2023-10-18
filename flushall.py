from redis import Redis

import sqlite3


redis_conn = Redis(host='localhost', port=6379, db=0)
redis_conn.flushall()



# Connect to the database (creates a new file if it doesn't exist)
conn = sqlite3.connect('image_features.db')
cursor = conn.cursor()

# Drop the table if it exists
cursor.execute('DROP TABLE IF EXISTS image_features')

# Create the new table
cursor.execute('''
CREATE TABLE image_features (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    face_features BLOB NOT NULL,
    dress_features BLOB NOT NULL
)
''')

conn.commit()
conn.close()

