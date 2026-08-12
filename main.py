"""
RoadGuard AI Backend
FastAPI application entry point.

Starts:
- MongoDB connection
- SimulationStore
- WeatherService
- Pothole detector
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env BEFORE importing/starting services that read environment variables.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.app_state import (
    connect_all,
    disconnect_all,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    """

    print("Starting RoadGuard backend services...")

    # Startup
    await connect_all()

    print("RoadGuard backend services started successfully.")

    try:
        yield
    finally:
        # Shutdown
        print("Stopping RoadGuard backend services...")
        await disconnect_all()
        print("RoadGuard backend services stopped.")


app = FastAPI(
    title="RoadGuard AI Backend",
    description="AI-powered pothole detection and rider safety backend",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Basic health endpoint
# ------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "RoadGuard AI Backend",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mongodb_uri_loaded": bool(os.getenv("MONGODB_URI")),
    }


# ------------------------------------------------------------------
# Application entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )