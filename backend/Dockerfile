FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install python-multipart

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]