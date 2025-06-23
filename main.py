import os
import tempfile
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Pydantic Models for Request Validation ---
class ProcessRequest(BaseModel):
    text: str
    options: dict

# --- Generic Model Loader ---
# This reusable function loads any local model based on your .env settings.
def load_local_llm(model_path_env_var, log_name):
    """Loads a single local GGUF model."""
    try:
        local_path = os.getenv(model_path_env_var)
        if not (local_path and os.path.exists(local_path)):
            logging.warning(f"🔥 {log_name} model path not found. Service will be disabled.")
            return None

        from ctransformers import AutoModelForCausalLM
        gpu_layers = int(os.getenv("GPU_LAYERS", 0))
        context_length = 4096

        logging.info(f"Loading {log_name} model (type: llama) from {local_path}...")
        return AutoModelForCausalLM.from_pretrained(
            local_path, model_type="llama", gpu_layers=gpu_layers, context_length=context_length
        )
    except Exception as e:
        logging.error(f"🔥 Critical error loading {log_name} model: {e}")
        return None

# --- Load AI Models ONCE on Startup ---
try:
    from faster_whisper import WhisperModel
    DEVICE = os.getenv("AI_DEVICE", "cpu")
    compute_type = "int8" if DEVICE == "cpu" else "float16"
    transcriber = WhisperModel(os.getenv("WHISPER_MODEL_NAME", "base"), device=DEVICE, compute_type=compute_type)
    logging.info(f"✅ Whisper model loaded successfully.")
except Exception as e:
    transcriber = None
    logging.error(f"🔥 Whisper could not be loaded: {e}")

# Load a single, powerful LLM for both text tasks.
llm = load_local_llm("SUMMARIZE_MODEL_PATH", "Text Generation")

# --- FastAPI Application ---
app = FastAPI(title="Laravel AI Companion API", version="1.1.0")

# --- API Endpoints ---
@app.get("/up", summary="Health Check", tags=["System"])
def health_check():
    return {"status": "online", "services": {"transcribe": bool(transcriber), "text_generation": bool(llm)}}

@app.post("/transcribe", summary="Transcribe audio", tags=["AI Services"])
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
    if not llm: raise HTTPException(503, "Text generation service is unavailable.")

    length = request.options.get('length', 'medium')
    # Use the Llama 3 prompt format
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSummarize the following text in a {length} paragraph:\n\n{request.text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    response_text = llm(prompt, max_new_tokens=1024, temperature=0.7)
    return {"text": response_text.strip(), "tokens": 0}

@app.post("/translate", summary="Translate text", tags=["AI Services"])
def translate_text(request: ProcessRequest):
    if not llm: raise HTTPException(503, "Text generation service is unavailable.")

    language = request.options.get('language', 'English')
    # Use the Llama 3 prompt format
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTranslate the following text to {language}:\n\n{request.text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    response_text = llm(prompt, max_new_tokens=1024, temperature=0.2)
    return {"text": response_text.strip(), "tokens": 0}
