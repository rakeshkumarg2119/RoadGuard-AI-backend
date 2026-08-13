"""
RoadGuard AI Backend — FastAPI entry point
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from core.app_state import connect_all, disconnect_all
from core.live_feed import live_feed
from routes.upload import router as upload_router
from routes.alert import router as alert_router
from routes.simulate import router as simulate_router

DASHBOARD_PATH = Path(__file__).parent / "core" / "dashboard.html"

# path fragment -> (stats key, label shown on the ticket)
CATEGORY_MAP = (
    ("/upload", "detection", "DETECTION"),
    ("/alert", "alert", "ALERT"),
    ("/simulate", "simulate", "SIMULATE"),
)

# paths we don't want cluttering the ops feed
FEED_EXCLUDE = {"/", "/ws/live", "/favicon.ico"}


def categorize(path: str):
    for fragment, key, label in CATEGORY_MAP:
        if fragment in path:
            return key, label
    return "other", "SYSTEM"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RoadGuard backend services...")
    await connect_all()
    print("RoadGuard backend ready.")
    try:
        yield
    finally:
        print("Stopping RoadGuard backend services...")
        await disconnect_all()
        print("Stopped.")


app = FastAPI(
    title="RoadGuard AI Backend",
    description="AI-powered pothole detection and rider safety backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def live_feed_middleware(request: Request, call_next):
    """Records every request (method, path, status, timing) and streams it
    to any open dashboard tabs. Doesn't touch request/response bodies, so
    it's safe with file uploads and won't slow anything down meaningfully."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    path = request.url.path
    if path not in FEED_EXCLUDE:
        key, label = categorize(path)
        await live_feed.record({
            "id": str(uuid.uuid4()),
            "time": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "category": key,
            "category_label": label,
            "client": request.client.host if request.client else "unknown",
        })
    return response


# ── Routes ────────────────────────────────────────────────────────
app.include_router(upload_router,   tags=["Detection"])
app.include_router(alert_router,    tags=["Alert"])
app.include_router(simulate_router, tags=["Simulate"])


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Live ops dashboard — shows every detection/alert/simulate request
    as it happens, for anyone with the server URL open in a browser."""
    return DASHBOARD_PATH.read_text(encoding="utf-8")


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await live_feed.connect(websocket)
    try:
        while True:
            # dashboard doesn't send anything meaningful; just keep the
            # connection open and detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        await live_feed.disconnect(websocket)


@app.get("/api/status")
async def api_status():
    return {"status": "ok", "service": "RoadGuard AI Backend", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "mongodb_uri_loaded": bool(os.getenv("MONGODB_URI")),
        "openweather_key_loaded": bool(os.getenv("OPENWEATHER_API_KEY")),
    }


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    # Give uvicorn a moment to actually start listening before we open the tab.
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)