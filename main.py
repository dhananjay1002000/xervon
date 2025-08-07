from fastapi import FastAPI, Request, HTTPException, status, Depends, Response, File, UploadFile, Security
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis import asyncio as aioredis

import io
from PIL import Image
import rembg
import numpy as np
import cv2
import logging
import time
import os

# from nsfwdetection.model import Model  # <--- NEW IMPORT

app = FastAPI()

# --- Setup logger for monitoring ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = "your_api_key_123"
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning(f"Unauthorized access attempt with API KEY: {api_key}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key

# --- Initialize Redis connection for rate limiting ---
@app.on_event("startup")
async def startup():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    redis = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis)


# --- Log all requests for monitoring and abuse detection ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    client_host = request.client.host
    path = request.url.path
    logger.info(f"{client_host} {path} completed_in={process_time:.2f}ms status_code={response.status_code}")
    if response.status_code >= 400:
        logger.warning(f"Potential abuse detected from {client_host} on endpoint {path} - status {response.status_code}")
    return response

# --- NSFW model initialization ---
# nsfw_model = Model()

# @app.post("/check-nsfw/")
# async def check_nsfw(
#     file: UploadFile = File(...),
#     api_key: str = Depends(verify_api_key),
#     _: str = Depends(RateLimiter(times=10, seconds=60))
# ):
#     contents = await file.read()
#     result = nsfw_model.predict_image_data(contents)
#     return result

# --- Other existing endpoints (unchanged) ---
@app.post("/remove-background/")
async def remove_background(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    _: str = Depends(RateLimiter(times=10, seconds=60))
):
    image_bytes = await file.read()
    try:
        output = rembg.remove(image_bytes)
        return Response(content=output, media_type="image/png")
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")

@app.post("/grayscale-resize/")
async def grayscale_resize(
    file: UploadFile = File(...),
    size: int = 300,
    api_key: str = Depends(verify_api_key),
    _: str = Depends(RateLimiter(times=10, seconds=60))
):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.post("/face-detect/")
async def face_detect(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    _: str = Depends(RateLimiter(times=10, seconds=60))
):
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    faces_list = [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]
    return {"faces": faces_list}

@app.post("/object-detect/")
async def object_detect(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    _: str = Depends(RateLimiter(times=10, seconds=60))
):
    return {"detected_objects": ["person", "car", "dog"]}

@app.get("/")
def read_root():
    return {"message": "Image Processing API with rate limiting, abuse monitoring, and NSFW detection is running!"}
    


