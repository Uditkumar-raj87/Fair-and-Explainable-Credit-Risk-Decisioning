FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & exec streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501"]