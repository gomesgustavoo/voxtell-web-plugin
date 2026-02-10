import os
# Set CUDA alloc conf to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import shutil
import tempfile
from typing import Annotated

import nibabel as nib
import numpy as np
import torch
import gc
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell.inference.predictor import VoxTellPredictor

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models/voxtell_v1.1")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading VoxTell model from {MODEL_DIR} on {DEVICE}...")
try:
    predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
    # Optimization: move prediction to CPU to save VRAM
    predictor.perform_everything_on_device = False
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

@app.post("/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()]
):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, image.filename)
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        print(f"Processing {image.filename} with prompt: '{prompt}'")

        try:
            # Force GC before inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load image using nnUNet's IO to handle reorientation if needed
            # predictable behavior with nibabel
            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([input_path])
            
            # Run inference
            # segmentations shape: (num_prompts, X, Y, Z)
            segmentations = predictor.predict_single_image(img, [prompt])
            
            # We only have one prompt, so take the first result
            seg_result = segmentations[0]
            
            # Save output
            output_filename = f"segmentation_{image.filename}"
            output_path = os.path.join(temp_dir, output_filename)
            
            reader_writer.write_seg(seg_result, output_path, props)
            
            # Save to a persistent temp location to return it
            system_tmp = tempfile.gettempdir()
            final_output_path = os.path.join(system_tmp, f"voxtell_output_{os.urandom(8).hex()}.nii.gz")
            shutil.copy(output_path, final_output_path)
            
            return FileResponse(
                final_output_path, 
                media_type="application/gzip", 
                filename=output_filename,
                background=None
            )

        except Exception as e:
            print(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
