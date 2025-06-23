import os
import tempfile
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel

# --- Configuration & Logging ---
# Sets up clean, timestamped logging for your API.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
load_dotenv() # Loads variables from your python_api/.env file

# --- Pydantic Models for Request Validation ---
# This ensures incoming data has the correct structure, like a Laravel Form Request.
class ProcessRequest(BaseModel):
    text: str
    options: dict

# --- Generic Model Loader ---
# This reusable function loads any local model based on your .env settings.
def load_model(local_path_env_var, model_type, log_name, context_length=4096):
    """Generic function to load a local GGUF model."""
    try:
        local_path = os.getenv(local_path_env_var)
        if local_path and os.path.exists(local_path):
            from ctransformers import AutoModelForCausalLM
            gpu_layers = int(os.getenv("GPU_LAYERS", 0))
            logging.info(f"Loading {log_name} model (type: {model_type}) from {local_path}...")
            logging.info(f"Using context_length={context_length}, gpu_layers={gpu_layers}")

            # This is the core model loading logic from the ctransformers library.
            return AutoModelForCausalLM.from_pretrained(
                local_path,
                model_type=model_type,
                gpu_layers=gpu_layers,
                context_length=context_length
            )
        else:
            logging.warning(f"🔥 {log_name} model path not found or invalid. Service will be disabled.")
            return None
    except Exception as e:
        logging.error(f"🔥 Critical error loading {log_name} model: {e}")
        return None

# --- Load AI Models ONCE on Startup ---
# This section runs only when the API starts, loading models into memory for fast responses.
try:
    from faster_whisper import WhisperModel
    # Reads the device setting from your .env file (cpu on VPS, mps on Mac).
    DEVICE = os.getenv("AI_DEVICE", "cpu")
    compute_type = "int8" if DEVICE == "cpu" else "float16"
    transcriber = WhisperModel(os.getenv("WHISPER_MODEL_NAME", "base"), device=DEVICE, compute_type=compute_type)
    logging.info(f"✅ Whisper model '{os.getenv('WHISPER_MODEL_NAME')}' loaded on device '{DEVICE}'.")
except Exception as e:
    transcriber = None
    logging.error(f"🔥 Whisper could not be loaded: {e}")

# FIX: Load BOTH models with the 'llama' type. This is the key to fixing the loading error.
# The model file may have 'qwen' in the name, but its GGUF architecture is Llama-based.
translator = load_model("TRANSLATE_MODEL_PATH", "llama", "Translator")
summarizer = load_model("SUMMARIZE_MODEL_PATH", "llama", "Summarizer")

# --- FastAPI Application ---
app = FastAPI(
    title="Laravel AI Companion API",
    version="1.0.0",
    description="A self-hosted API for transcription, summarization, and translation."
)

# --- API Endpoints ---
@app.get("/up", summary="Health Check", tags=["System"])
def health_check():
    """Checks if the API is running and which AI services are loaded successfully."""
    return {
        "status": "online",
        "services": {
            "transcribe": bool(transcriber),
            "translate": bool(translator),
            "summarize": bool(summarizer)
        }
    }

@app.post("/transcribe", summary="Transcribe an audio file", tags=["AI Services"])
async def transcribe_audio(file: UploadFile = File(...)):
    """Accepts an audio file and returns the transcribed text."""
    if not transcriber:
        raise HTTPException(status_code=503, detail="Transcription service is unavailable.")
    try:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(await file.read())
            temp_file.flush()
            segments, _ = transcriber.transcribe(temp_file.name, beam_size=5)
            return {"transcript": " ".join(s.text for s in segments).strip()}
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio file.")

@app.post("/summarize", summary="Summarize text", tags=["AI Services"])
def summarize_text(request: ProcessRequest):
    """Accepts text and returns a summary based on the TowerBase model."""
    if not summarizer:
        raise HTTPException(status_code=503, detail="Summarization service is unavailable.")

    length = request.options.get('length', 'medium')
    # This Llama-style prompt is correct for the TowerBase model.
    prompt = f"<|user|>\nSummarize the following text in a {length} paragraph:\n\n{request.text}<|end|>\n<|assistant|>\n"
    summary = summarizer(prompt, max_new_tokens=1024, temperature=0.7)

    return {"text": summary.strip(), "tokens": 0}

@app.post("/translate", summary="Translate text", tags=["AI Services"])
def translate_text(request: ProcessRequest):
    """Accepts text and returns a translation based on the Sanskrit/Qwen model."""
    if not translator:
        raise HTTPException(status_code=503, detail="Translation service is unavailable.")

    language = request.options.get('language', 'English')

    # FIX: Use the Llama-style prompt format, which is required when loading with model_type='llama'.
    prompt = f"<|user|>\nTranslate the following text to {language}:\n\n{request.text}<|end|>\n<|assistant|>\n"

    translation = translator(prompt, max_new_tokens=1024, temperature=0.2)

    return {"text": translation.strip(), "tokens": 0}
