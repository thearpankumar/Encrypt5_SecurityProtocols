# Setup Guide for Flask and PostgreSQL in Docker

This guide explains how to set up a Dockerized Flask application and PostgreSQL database. It includes commands to create tables, insert data, and interact with the API.

## Prerequisites
1. Docker and Docker Compose installed on your machine.
2. Basic knowledge of Python and SQL.

---

## Step 1: Create the `docker-compose.yml` File

```yaml
version: '3.9'
services:
  flask-server:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_HOST=postgres
      - DATABASE_PORT=5432
      - DATABASE_USER=your_username
      - DATABASE_PASSWORD=your_password
      - DATABASE_NAME=your_database_name
    depends_on:
      - postgres
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_USER: your_username
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: your_database_name
    ports:
      - "5432:5432"
```

---

## Step 2: Create the Flask Application

1. Add the Flask server code to `app.py`.
2. Create a `Dockerfile` for Flask:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

3. Create `requirements.txt`:

```
Flask
psycopg2
```

---

## Step 3: Start the Containers

Run the following command to build and start the Docker containers:

```bash
docker-compose up --build
```

---

## Step 4: Connect to PostgreSQL

1. Access the PostgreSQL container:

```bash
docker exec -it <postgres_container_id> psql -U your_username -d your_database_name
```

2. Create the necessary tables:

```sql
CREATE TABLE FileFolderMetadata (
    ID UUID PRIMARY KEY,
    ParentID UUID,
    UserID UUID,
    Name VARCHAR NOT NULL,
    Type VARCHAR NOT NULL,
    Uploaded TIMESTAMP NOT NULL,
    Accessed TIMESTAMP,
    FileSize INTEGER
);
```

---

## Step 5: Insert Dummy Data

Run the following SQL commands to insert dummy data into `FileFolderMetadata`:

```sql
INSERT INTO FileFolderMetadata (ID, ParentID, UserID, Name, Type, Uploaded, Accessed, FileSize)
VALUES
('21111111-2222-3333-4444-555555555555', NULL, '11111111-2222-3333-4444-555555555555', 'Documents', 'folder', '2024-12-13 10:05:00', '2024-12-13 10:05:00', NULL),
('31111111-2222-3333-4444-555555555555', '21111111-2222-3333-4444-555555555555', '11111111-2222-3333-4444-555555555555', 'Resume.pdf', 'document', '2024-12-13 10:10:00', '2024-12-13 10:10:00', 2048);
```

---

## Step 6: Test the API

Use the following `curl` command to test the API:

```bash
curl -X GET http://localhost:5000/filefoldermetadata
```

This should return the data in JSON format:

```json
[
  {
    "ID": "21111111-2222-3333-4444-555555555555",
    "ParentID": null,
    "UserID": "11111111-2222-3333-4444-555555555555",
    "Name": "Documents",
    "Type": "folder",
    "Uploaded": "2024-12-13T10:05:00",
    "Accessed": "2024-12-13T10:05:00",
    "FileSize": null
  },
  {
    "ID": "31111111-2222-3333-4444-555555555555",
    "ParentID": "21111111-2222-3333-4444-555555555555",
    "UserID": "11111111-2222-3333-4444-555555555555",
    "Name": "Resume.pdf",
    "Type": "document",
    "Uploaded": "2024-12-13T10:10:00",
    "Accessed": "2024-12-13T10:10:00",
    "FileSize": 2048
  }
]
```

---

## Additional Commands

### Stop Containers
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f flask-server
```

---

This completes the setup and testing of the Dockerized Flask application with PostgreSQL.

