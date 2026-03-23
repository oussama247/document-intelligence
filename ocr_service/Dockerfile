FROM python:3.10

RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]