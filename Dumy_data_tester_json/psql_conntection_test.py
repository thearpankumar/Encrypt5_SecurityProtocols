import psycopg2

def connect_to_db(dbname, user, password, host, port):
    try:
        conn = psycopg2.connect(
            database=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connected to database successfully!")
        return conn
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

# Example usage:
conn = connect_to_db(
    dbname="ecrypt5_main",
    user="postgres",
    password="encrypt5",
    host="localhost",
    port="5432"
)

# Once connected, you can execute SQL queries:
cur = conn.cursor()
#cur.execute("SELECT * FROM filefoldermetadata")
cur.execute("SELECT * FROM filefoldermetadata WHERE type = 'music';")
rows = cur.fetchall()
for row in rows:
    print(row)

cur.close()
conn.close()
