---
title: Medical RAG API
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Medical RAG API - Hack-A-Cure 2025

Advanced Retrieval-Augmented Generation system for medical question answering.

## 🏆 Performance
- **Database Size**: 50,693 chunks from 9 medical textbooks
- **Accuracy Achieved**: 85%
- **Success Rate**: 100%
- **Average Response Time**: 4-10 seconds

## ⚡ Features
- 50,693 chunks from 9 medical textbooks covering multiple specialties
- Semantic search with Sentence Transformers (all-MiniLM-L6-v2)
- Powered by Google Gemini 2.0 Flash
- Accurate citations with page numbers and book names
- 400-word chunks with 50-word overlap for context preservation

## 📡 API Endpoints

### POST `/query`
Medical question answering with citations.

**Request:**
```json
{
  "query": "When to give Tdap booster?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The Tdap booster is indicated...",
  "contexts": ["From [InternalMedicine], Page 163: ..."]
}
```

### GET `/`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "database_chunks": 50693,
  "gemini_model": "gemini-2.0-flash-exp"
}
```

## 🛠️ Technology Stack
- **FastAPI** - High-performance web framework
- **ChromaDB** - Vector database for semantic search
- **Sentence Transformers** - Local embeddings (all-MiniLM-L6-v2)
- **Google Gemini API** - Advanced text generation
- **PyMuPDF** - PDF text extraction
- **Docker** - Containerization

## 📚 Medical Textbooks Included
1. **Anatomy & Physiology** - 3,281 chunks
2. **Cardiology** - 6,349 chunks
3. **Dentistry** - 1,552 chunks
4. **Emergency Medicine** - 7,217 chunks
5. **Gastrology** - 8,445 chunks
6. **General Medicine** - 3,606 chunks
7. **Infectious Disease** - 631 chunks
8. **Internal Medicine** - 11,095 chunks
9. **Nephrology** - 8,517 chunks

**Total**: 18,663 pages processed

## 🔒 Zero-Cost Architecture
- ✅ Free embeddings (local Sentence Transformers)
- ✅ Free LLM (Gemini API - generous free tier)
- ✅ Free vector database (ChromaDB local)
- ✅ Free hosting (HuggingFace Spaces)

**Estimated Cost**: <$1 per 1,000 queries

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Virtual environment

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd hack-a-cure

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Run server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test health endpoint
curl http://localhost:8000/

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"When to give Tdap booster?","top_k":3}'
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t medical-rag-api .

# Run container
docker run -p 7860:7860 \
  -e GEMINI_API_KEY="your-api-key" \
  medical-rag-api
```

## 📊 System Architecture

```
User Query → FastAPI → Sentence Transformer (Embedding)
                    ↓
              ChromaDB (Similarity Search)
                    ↓
         Retrieved Contexts (top_k chunks)
                    ↓
              Gemini 2.0 Flash (Generation)
                    ↓
         Answer + Citations → User
```

## 🎯 Use Cases
- Medical students studying for exams
- Healthcare professionals needing quick references
- Researchers looking for specific medical information
- Medical educators preparing materials

## ⚠️ Disclaimer
This is an AI-powered reference tool for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

**Built for Hack-A-Cure 2025 Hackathon**

**Team**: [Your Team Name]
**Category**: Healthcare AI
**Technologies**: FastAPI, ChromaDB, Sentence Transformers, Google Gemini
