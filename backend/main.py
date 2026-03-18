import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .routers import matrix, progress

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Capstone Matrix Multiplication Client")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.debug(
        f"Request: {request.method} {request.url} "
        f"Origin={request.headers.get('origin')} "
        f"Content-Type={request.headers.get('content-type')} "
        f"Body={body[:500] if body else b'(empty)'}"
    )
    response = await call_next(request)
    logger.debug(f"Response: {response.status_code} for {request.method} {request.url}")
    return response


app.include_router(matrix.router)
app.include_router(progress.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
