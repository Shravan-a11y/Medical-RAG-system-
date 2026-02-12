FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY app.py .

# Copy ChromaDB database
COPY chroma_db ./chroma_db

# Expose HuggingFace port
EXPOSE 7860

# Environment variable for Gemini API key
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
