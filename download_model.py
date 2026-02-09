from huggingface_hub import snapshot_download

MODEL_NAME = "voxtell_v1.1" # Updated models may be available in the future
DOWNLOAD_DIR = "./models/" # Optionally specify the download directory

download_path = snapshot_download(
      repo_id="mrokuss/VoxTell",
      allow_patterns=[f"{MODEL_NAME}/*", "*.json"],
      local_dir=DOWNLOAD_DIR
)

# path to model directory, e.g., "/home/user/temp/voxtell_v1.1"
model_path = f"{download_path}/{MODEL_NAME}"