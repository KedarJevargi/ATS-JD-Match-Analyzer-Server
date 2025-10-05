import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# Import your routers
from routers import pdf_router, jd_router, ats_router

PORT = int(os.getenv("PORT", 8000))
app = FastAPI(
    title="My Backend API",
    description="A simple FastAPI service with PDF text extraction",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ats-jd-match-analyzer-client.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Server is running 🚀"}

# Register routers
app.include_router(pdf_router.router)
app.include_router(jd_router.router)
app.include_router(ats_router.router)

