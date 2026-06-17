FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kaynak kod
COPY src/ ./src/
COPY tests/ ./tests/
COPY config.yaml .

CMD ["pytest", "tests/", "-v"]