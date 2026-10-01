# The image Cloudflare runs. wrangler builds it locally with Docker on
# `wrangler deploy` and pushes it to Cloudflare's registry.

# slim, not full: a smaller image boots faster, and boot time is the cold start.
# No --platform here: wrangler already builds for linux/amd64.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, code second, so editing main.py doesn't reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Must match defaultPort in the Container class (src/index.js).
EXPOSE 8080

# 0.0.0.0, not 127.0.0.1: requests arrive from outside the container's loopback.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
