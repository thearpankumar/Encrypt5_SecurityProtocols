import psycopg2
from psycopg2.extras import execute_batch
import uuid
import random
import datetime

# Database connection configuration
db_config = {
    'host': 'localhost',  # Change as needed
    'port': 5432,
    'database': 'ecrypt5_main',
    'user': 'postgres',
    'password': 'encrypt5'
}

# Function to fetch all user IDs from the rootfolders table
def fetch_user_ids():
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        query = "SELECT userid FROM rootfolders"
        cursor.execute(query)
        user_ids = [row[0] for row in cursor.fetchall()]

        cursor.close()
        connection.close()

        if not user_ids:
            raise ValueError("No user IDs found in rootfolders table.")

        return user_ids
    except Exception as e:
        print(f"Error fetching user IDs: {e}")
        return []

# Function to generate random data for a parent row
def generate_parent_row(user_ids):
    return (
        str(uuid.uuid4()),  # id
        None,               # parentid
        random.choice(user_ids),  # userid
        f"Folder_{random.randint(1, 1000000)}",  # name
        'folder',           # type
        datetime.datetime.now().isoformat(),  # uploaded
        datetime.datetime.now().isoformat() if random.random() > 0.5 else None,  # accessed
        None                # filesize
    )

# Function to generate random data for a child row
def generate_child_row(parent_ids, user_ids):
    return (
        str(uuid.uuid4()),  # id
        random.choice(parent_ids),  # parentid
        random.choice(user_ids),  # userid
        f"File_{random.randint(1, 1000000)}",  # name
        random.choice(['document', 'photo', 'music', 'video', 'other']),  # type
        datetime.datetime.now().isoformat(),  # uploaded
        datetime.datetime.now().isoformat() if random.random() > 0.5 else None,  # accessed
        random.randint(0, 102400) if random.random() > 0.2 else None  # filesize
    )

# Function to insert parent rows and retrieve their IDs
def insert_parents_and_get_ids(total_parents, batch_size, user_ids):
    parent_ids = []
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        query = """
        INSERT INTO filefoldermetadata (id, parentid, userid, name, type, uploaded, accessed, filesize)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
        """

        for _ in range(0, total_parents, batch_size):
            rows = [generate_parent_row(user_ids) for _ in range(batch_size)]
            for row in rows:
                cursor.execute(query, row)
                parent_ids.append(cursor.fetchone()[0])

        connection.commit()
        cursor.close()
        connection.close()

        print(f"Inserted {total_parents} parent rows successfully.")
    except Exception as e:
        print(f"Error during parent row insertion: {e}")

    return parent_ids

# Function to insert child rows
def insert_child_rows(total_children, batch_size, parent_ids, user_ids):
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        query = """
        INSERT INTO filefoldermetadata (id, parentid, userid, name, type, uploaded, accessed, filesize)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for _ in range(0, total_children, batch_size):
            rows = [generate_child_row(parent_ids, user_ids) for _ in range(batch_size)]
            execute_batch(cursor, query, rows)

        connection.commit()
        cursor.close()
        connection.close()

        print(f"Inserted {total_children} child rows successfully.")
    except Exception as e:
        print(f"Error during child row insertion: {e}")

if __name__ == "__main__":
    TOTAL_PARENT_ROWS = 10000  # Total number of parent rows (folders) to insert
    TOTAL_CHILD_ROWS = 990000  # Total number of child rows (files) to insert
    BATCH_SIZE = 1000          # Number of rows per batch

    # Fetch user IDs from rootfolders table
    user_ids = fetch_user_ids()

    if not user_ids:
        print("No user IDs found. Ensure the rootfolders table is populated.")
    else:
        # Step 1: Insert parent rows and retrieve their IDs
        parent_ids = insert_parents_and_get_ids(TOTAL_PARENT_ROWS, BATCH_SIZE, user_ids)

        # Step 2: Insert child rows using the retrieved parent IDs
        insert_child_rows(TOTAL_CHILD_ROWS, BATCH_SIZE, parent_ids, user_ids)
