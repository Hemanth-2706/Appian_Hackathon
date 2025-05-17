from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Allow frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve local image files
app.mount("/images", StaticFiles(directory="Images"), name="images")

@app.post("/recommend")
async def recommend(file: UploadFile = File(None), query: str = Form(None)):
    # [✓] Dummy logic: return first 5 images in folder
    image_dir = "Images"
    image_paths = sorted([
        f"/images/{img}" for img in os.listdir(image_dir)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    recommendations = image_paths[:5]
    return JSONResponse(content={"results": recommendations})
