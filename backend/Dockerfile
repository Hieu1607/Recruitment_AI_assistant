FROM python:3.11-slim

WORKDIR /app

# Enable bytecode compilation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc libpq-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
