Medical RAG API - Hack-A-Cure 2025

Advanced Retrieval-Augmented Generation system for medical question answering.

🏆 Performance
- Database Size: 50,693 chunks from 9 medical textbooks
- Accuracy Achieved: 85%
- Success Rate: 100%
- Average Response Time: 4-10 seconds

⚡ Features
- 50,693 chunks from 9 medical textbooks covering multiple specialties
- Semantic search with Sentence Transformers (all-MiniLM-L6-v2)
- Powered by Google Gemini 2.0 Flash
- Accurate citations with page numbers and book names
- 400-word chunks with 50-word overlap for context preservation

📡 API Endpoints

 POST `/query`
Medical question answering with citations.


🛠️ Technology Stack
- **FastAPI** - High-performance web framework
- **ChromaDB** - Vector database for semantic search
- **Sentence Transformers** - Local embeddings (all-MiniLM-L6-v2)
- **Google Gemini API** - Advanced text generation
- **PyMuPDF** - PDF text extraction
- **Docker** - Containerization

📚 Medical Textbooks Included
1. Anatomy & Physiology - 3,281 chunks
2. Cardiology - 6,349 chunks
3. Dentistry- 1,552 chunks
4. Emergency Medicine - 7,217 chunks
5. Gastrology - 8,445 chunks
6. General Medicine - 3,606 chunks
7. Infectious Disease - 631 chunks
8. Internal Medicine - 11,095 chunks
9. Nephrology- 8,517 chunks

Total: 18,663 pages processed

🔒 Zero-Cost Architecture
- ✅ Free embeddings (local Sentence Transformers)
- ✅ Free LLM (Gemini API - generous free tier)
- ✅ Free vector database (ChromaDB local)
- ✅ Free hosting (HuggingFace Spaces)

Estimated Cost: <$1 per 1,000 queries

📊 System Architecture


User Query → FastAPI → Sentence Transformer (Embedding)
                    ↓
              ChromaDB (Similarity Search)
                    ↓
         Retrieved Contexts (top_k chunks)
                    ↓
              Gemini 2.0 Flash (Generation)
                    ↓
         Answer + Citations → User


 Use Cases
- Medical students studying for exams
- Healthcare professionals needing quick references
- Researchers looking for specific medical information
- Medical educators preparing materials

Disclaimer
This is an AI-powered reference tool for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.


Team: Mirage
Category: Healthcare AI
Technologies: FastAPI, ChromaDB, Sentence Transformers, Google Gemini
