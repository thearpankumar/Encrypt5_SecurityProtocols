from flask import Flask, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

# Database connection configuration
db_config = {
    'host': 'localhost',  # Change to your Docker container hostname if different
    'port': 5432,         # Default PostgreSQL port
    'database': 'ecrypt5_main',
    'user': 'postgres',
    'password': 'encrypt5'
}

@app.route('/filefoldermetadata', methods=['GET'])
def get_filefoldermetadata():
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        # Query to fetch all data from FileFolderMetadata
        query = "SELECT * FROM FileFolderMetadata;"
        cursor.execute(query)
        results = cursor.fetchall()

        # Return the results in JSON format
        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Ensure the connection is closed
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
