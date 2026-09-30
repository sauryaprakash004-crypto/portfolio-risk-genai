FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Runs the full analysis; then e.g.: docker run ... python ask.py "Explain my portfolio risk."
# ANTHROPIC_API_KEY is optional: without it the offline explanation templates are used.
CMD ["python", "run_analysis.py"]
