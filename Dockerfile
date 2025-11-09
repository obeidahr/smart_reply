# ---- Base ----
FROM python:3.10-slim

# ---- Setup ----
WORKDIR /app
COPY requirements.txt .

# Install dependencies (faster with no cache and single layer)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# ---- Environment ----
EXPOSE 8000

# ---- Run ----
CMD ["python", "main.py"]
