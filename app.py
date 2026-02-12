import os
import asyncio
from typing import List, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Chatbot API",
    description="Query medical textbooks using Retrieval Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Request and Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The medical question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of contexts to retrieve")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer from Gemini")
    contexts: List[str] = Field(..., description="Retrieved context chunks")


# Global variables for models and database
embedding_model: Optional[SentenceTransformer] = None
chroma_client: Optional[chromadb.PersistentClient] = None
chroma_collection = None
genai_configured: bool = False


def initialize_models():
    """Initialize embedding model, ChromaDB, and Gemini API"""
    global embedding_model, chroma_client, chroma_collection, genai_configured
    
    # Load embedding model
    print("Loading embedding model: all-MiniLM-L6-v2...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")
    
    # Initialize ChromaDB
    db_path = Path("./chroma_db")
    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {db_path.absolute()}. "
            "Please run preprocess_textbooks.py first."
        )
    
    print("Loading ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        chroma_collection = chroma_client.get_collection(name="medical_textbooks")
        count = chroma_collection.count()
        print(f"✓ ChromaDB loaded with {count} chunks")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ChromaDB collection. Error: {str(e)}. "
            "Please run preprocess_textbooks.py first."
        )
    
    # Configure Gemini API
    api_key = "Place your Gemini API key"
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with your Gemini API key."
        )
    
    print("Configuring Gemini API...")
    genai.configure(api_key=api_key)
    genai_configured = True
    print("✓ Gemini API configured")


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        initialize_models()
        print("\n✓ API is ready to handle requests")
    except Exception as e:
        print(f"\n✗ Startup failed: {str(e)}")
        raise


@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    if not embedding_model or not chroma_collection or not genai_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not fully initialized"
        )
    
    chunk_count = chroma_collection.count()
    
    return {
        "status": "healthy",
        "message": "Medical RAG API is running",
        "database_chunks": chunk_count,
        "gemini_model": "models/gemini-2.5-flash"
    }


def retrieve_contexts(query: str, top_k: int) -> List[dict]:
    """
    Retrieve relevant contexts from ChromaDB
    
    Args:
        query: The query string
        top_k: Number of results to retrieve
        
    Returns:
        List of context dictionaries with metadata
        
    Raises:
        RuntimeError: If embedding generation or ChromaDB query fails
    """
    try:
        print(f"[DEBUG] Generating embedding for query: {query}")
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        print("[DEBUG] Successfully generated embedding")
        
        # Query ChromaDB with enhanced retrieval
        print(f"[DEBUG] Querying ChromaDB with top_k={top_k}")
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k + 2,  # Get extra results for better context
            where={"page_number": {"$gt": 0}},  # Ensure we get valid page numbers
            include=["metadatas", "documents", "distances"]
        )
        
        # Filter results based on relevance score
        if results and results.get('distances') and len(results['distances']) > 0:
            distances = results['distances'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # Enhanced filtering for better context quality
            filtered_results = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Skip if distance is too high (less relevant)
                if dist >= 1.0:
                    continue
                    
                # Skip very short contexts
                if len(doc.split()) < 20:  # Skip contexts with fewer than 20 words
                    continue
                    
                # Skip contexts without clear medical content
                medical_keywords = ['treatment', 'diagnosis', 'symptoms', 'disease', 'patient', 
                                'clinical', 'medical', 'therapy', 'condition', 'health',
                                'drug', 'medication', 'procedure', 'protocol', 'care']
                if not any(keyword in doc.lower() for keyword in medical_keywords):
                    continue
                    
                filtered_results.append((doc, meta, dist))
            
            # Sort by relevance and take top_k
            filtered_results.sort(key=lambda x: x[2])  # Sort by distance
            filtered_results = filtered_results[:top_k]  # Take only top_k most relevant
            
            # Reconstruct results
            results = {
                'documents': [[item[0] for item in filtered_results]],
                'metadatas': [[item[1] for item in filtered_results]]
            }
        print("[DEBUG] Successfully queried ChromaDB")
        
    except Exception as e:
        print(f"[ERROR] Failed to retrieve contexts: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to retrieve contexts: {str(e)}")
    
    # Format results
    contexts = []
    if results and results['documents'] and len(results['documents']) > 0:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        for doc, meta in zip(documents, metadatas):
            contexts.append({
                'text': doc,
                'book_name': meta.get('book_name', 'Unknown'),
                'page_number': meta.get('page_number', 0)
            })
    
    print(f"[DEBUG] Formatted {len(contexts)} contexts")
    return contexts


def format_contexts(contexts: List[dict]) -> List[str]:
    """
    Format contexts as plain strings with citations
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        List of formatted context strings
    """
    formatted = []
    for ctx in contexts:
        formatted_str = f"From [{ctx['book_name']}], Page {ctx['page_number']}: {ctx['text']}"
        formatted.append(formatted_str)
    
    return formatted


async def generate_answer_with_timeout(query: str, contexts: List[str], timeout: int = 60) -> str:
    """
    Generate answer using Gemini API with timeout
    
    Args:
        query: The user's question
        contexts: List of formatted context strings
        timeout: Timeout in seconds (default 60)
        
    Returns:
        Generated answer string
        
    Raises:
        asyncio.TimeoutError: If generation exceeds timeout
        ValueError: If Gemini API returns empty response
        RuntimeError: For other API-related errors
    """
    async def generate():
        try:
            # Prepare the prompt with contexts
            context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
            
            prompt = f"""You are an expert medical professional with extensive clinical experience. Format your response using markdown for better readability. Answer the following medical question using the provided medical textbook contexts.

Question: {query}

Available Medical Literature:
{context_text}

Instructions for Response Format:

Format your response using this structure and markdown:

## Summary Answer
- Begin with a clear, concise summary (2-3 sentences)
- Use bullet points for key takeaways

## Detailed Explanation
- Break down the topic into clear sections
- Use **bold** for important medical terms
- Use subheadings (###) for different aspects

## Mechanism of Action
- Explain physiological/pathological processes
- Use clear bullet points
- Include citations [Book Name, Page X]

### Clinical Implications
- List key clinical considerations
- Include treatment approaches
- Highlight important warnings

### Key Terms
> Create a box with definitions of technical terms

## Evidence & Research
- Cite specific textbook evidence
- Use numbered lists for steps/procedures
- Include relevant statistical data if available

---
### 💡 Quick Reference
- Bullet point key takeaways
- Focus on practical applications
- List essential reminders

_Note: Include citations [Book Name, Page X] throughout your response where appropriate._

Additional Guidelines:
- Prioritize accuracy and evidence-based information
- Use proper medical terminology with explanations
- Structure information in a logical clinical sequence
- Support each major point with textbook citations
- Address limitations in available information when present
- Focus on practical, clinically relevant details
- Maintain professional medical tone throughout
- Include relevant contraindications or warnings

If certain aspects of the question cannot be fully addressed with the available contexts, acknowledge this limitation and focus on what can be accurately answered with the provided sources.

Answer:"""

            # Log prompt length for debugging
            print(f"[DEBUG] Sending request to Gemini API with prompt length: {len(prompt)}")
            
            # Generate response using Gemini
            model = genai.GenerativeModel('models/gemini-2.5-flash')
            response = await asyncio.to_thread(
                lambda: model.generate_content(prompt).text
            )
            
            print("[DEBUG] Received response from Gemini API")
            
            if not response:
                raise ValueError("Empty response from Gemini API")
            
            return response
            
        except Exception as e:
            print(f"[ERROR] Error in generate(): {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Gemini API error: {str(e)}")
    
    # Execute with timeout
    try:
        print(f"[DEBUG] Starting answer generation with {timeout}s timeout")
        answer = await asyncio.wait_for(generate(), timeout=timeout)
        print("[DEBUG] Successfully generated answer")
        return answer
    except asyncio.TimeoutError:
        print(f"[ERROR] Answer generation timed out after {timeout} seconds")
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in generate_answer_with_timeout: {str(e)}")
        raise


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint for medical questions
    
    Args:
        request: QueryRequest with query and top_k
        
    Returns:
        QueryResponse with answer and contexts
    """
    try:
        print(f"[DEBUG] Received query: {request.query}, top_k: {request.top_k}")
        
        # Validate models are loaded
        if not embedding_model:
            print("[ERROR] Embedding model not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding model not initialized"
            )
        if not chroma_collection:
            print("[ERROR] ChromaDB collection not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChromaDB collection not initialized"
            )
        if not genai_configured:
            print("[ERROR] Gemini API not configured")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Gemini API not configured"
            )
        
        print("[DEBUG] All models validated, retrieving contexts...")
        
        try:
            # Retrieve contexts from ChromaDB
            context_dicts = retrieve_contexts(request.query, request.top_k)
            print(f"[DEBUG] Retrieved {len(context_dicts)} contexts")
        except Exception as e:
            print(f"[ERROR] Failed to retrieve contexts: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve contexts: {str(e)}"
            )
        
        if not context_dicts:
            print("[WARNING] No contexts found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant contexts found in the database"
            )
        
        # Format contexts
        formatted_contexts = format_contexts(context_dicts)
        
        # Generate answer with Gemini (with 60-second timeout)
        try:
            answer = await generate_answer_with_timeout(
                request.query,
                formatted_contexts,
                timeout=60
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Answer generation timed out after 60 seconds"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating answer: {str(e)}"
            )
        
        # Return response
        return QueryResponse(
            answer=answer,
            contexts=formatted_contexts
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Medical RAG Chatbot API...")
    print("Make sure GEMINI_API_KEY environment variable is set!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )