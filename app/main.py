import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.config import settings

app = FastAPI(
    title="Document Intelligence Platform",
    description="Production-ready RAG system for intelligent document retrieval.",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router, prefix="/api", tags=["Documents"])

@app.get("/")
async def root():
    return {"status": "Online", "message": "Welcome to the Document Intelligence Platform API"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
