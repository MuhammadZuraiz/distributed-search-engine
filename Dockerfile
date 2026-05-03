FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/index logs data/crawled_distributed

CMD ["python", "search/app.py"]