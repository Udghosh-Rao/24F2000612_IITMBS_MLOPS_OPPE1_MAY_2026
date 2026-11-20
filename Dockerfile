FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_PATH=/app/models/model.joblib

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app.py .

# Mount or copy a trained bundle: -v $(pwd)/models/best_iteration_2.joblib:/app/models/model.joblib
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
