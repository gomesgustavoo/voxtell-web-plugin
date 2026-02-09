# Voxtell Integration

This document outlines the core components and run commands for the Voxtell medical image segmentation integration.

## Core Components

### Backend (`/backend`, `/voxtell`)
- **`backend/server.py`**: A FastAPI server that exposes the `/predict` endpoint.
  - Handles `.nii.gz` file uploads and text prompts.
  - Returns the segmentation mask as a GZIP-compressed NIfTI file.
  - **Key Optimizations**:
    - `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`: Reduces GPU memory fragmentation.
    - `predictor.perform_everything_on_device = False`: Offloads some sliding window operations to CPU to save VRAM.
- **`voxtell/inference/predictor.py`**: The inference engine.
  - **Optimization**: The text encoder (`Qwen/Qwen3-Embedding-4B`) is loaded in `float16` precision to reduce VRAM usage from ~15GB to ~7.5GB.

### Frontend (`/frontend`)
- **`src/App.tsx`**: The main application interface.
  - Manages file selection, prompt input, and API communication.
  - Displays processing status and errors.
- **`src/components/Viewer.tsx`**: A 3D NIfTI viewer using `@niivue/niivue`.
  - Renders the original scan and overlays the segmentation mask in red (50% opacity).

## Run Commands

### 1. Backend Server
Ensure the `voxtell` conda environment is activated.

```bash
# From the project root
conda activate voxtell
python backend/server.py
```
*The server runs on `http://0.0.0.0:8000`.*

### 2. Frontend Application
Run the Vite development server.

```bash
cd frontend
npm run dev
```
*Access the application at `http://localhost:5173`.*

## Usage
1. Open the frontend URL.
2. Click **"Choose File"** (or similar input) to upload a `.nii` or `.nii.gz` file.
3. Enter an anatomical prompt (e.g., "liver", "spleen", "right kidney").
4. Click **"Run Segmentation"**.
5. Wait for the result to appear in the viewer.
