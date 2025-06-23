import os
import tempfile
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Pydantic Models for Request Validation ---
class ProcessRequest(BaseModel):
    text: str
    options: dict

# --- Generic Model Loader ---
def load_model(provider_env_var, local_path_env_var, api_model_name_var, local_model_type, log_name):
    """Generic function to load a model based on the .env configuration."""
    provider = os.getenv(provider_env_var, "local").lower()

    try:
        if provider == "local":
            local_path = os.getenv(local_path_env_var)
            if local_path and os.path.exists(local_path):
                from ctransformers import AutoModelForCausalLM
                logging.info(f"Loading {log_name} model from local path: {local_path}...")
                # Offload all possible layers to GPU for best performance
                GPU_LAYERS = int(os.getenv("GPU_LAYERS", 0))
                return AutoModelForCausalLM.from_pretrained(local_path, model_type=local_model_type, gpu_layers=GPU_LAYERS)
            else:
                logging.warning(f"🔥 {log_name} provider set to 'local' but path not found. Service will be disabled.")
                return None
        elif provider == "api":
            from transformers import pipeline
            api_model_name = os.getenv(api_model_name_var)
            logging.info(f"Loading {log_name} model from Hugging Face API: {api_model_name}...")
            task = "translation_xx_to_yy" if log_name == "Translator" else "summarization"
            return pipeline(task, model=api_model_name)
        else:
            logging.warning(f"🔥 Invalid provider '{provider}' for {log_name}. Service will be disabled.")
            return None
    except Exception as e:
        logging.error(f"🔥 Critical error loading {log_name} model: {e}")
        return None

# --- Load AI Models ONCE on Startup ---
try:
    from faster_whisper import WhisperModel
    DEVICE = os.getenv("AI_DEVICE", "cpu")
    transcriber = WhisperModel(os.getenv("WHISPER_MODEL_NAME", "base"), device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "float16")
    logging.info(f"✅ Whisper model '{os.getenv('WHISPER_MODEL_NAME')}' loaded successfully.")
except Exception as e:
    transcriber = None; logging.error(f"🔥 Whisper could not be loaded: {e}")

translator = load_model("TRANSLATION_PROVIDER", "TRANSLATE_MODEL_PATH", "API_TRANSLATE_MODEL_NAME", "qwen", "Translator")
summarizer = load_model("SUMMARIZATION_PROVIDER", "SUMMARIZE_MODEL_PATH", "API_SUMMARIZE_MODEL_NAME", "llama", "Summarizer")

# --- FastAPI Application ---
app = FastAPI(title="Laravel AI Companion API", version="1.0.0")

# --- API Endpoints ---
@app.get("/up", summary="Health Check", tags=["System"])
def health_check():
    return {"status": "woke up", "services": {"transcribe": bool(transcriber), "translate": bool(translator), "summarize": bool(summarizer)}}

@app.post("/transcribe", summary="Transcribe an audio file", tags=["AI Services"])
async def transcribe_audio(file: UploadFile = File(...)):
    if not transcriber: raise HTTPException(503, "Transcription service is currently unavailable.")
    try:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(await file.read())
            segments, _ = transcriber.transcribe(temp_file.name, beam_size=5)
            return {"transcript": " ".join(s.text for s in segments).strip()}
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(500, "Failed to process audio file.")

@app.post("/summarize", summary="Summarize text", tags=["AI Services"])
def summarize_text(request: ProcessRequest):
    if not summarizer: raise HTTPException(503, "Summarization service is currently unavailable.")

    length = request.options.get('length', 'medium')
    prompt = f"<|user|>\nSummarize the following text in a {length} paragraph:\n\n{request.text}<|end|>\n<|assistant|>\n"
    summary = summarizer(prompt, max_new_tokens=1024, temperature=0.7)

    return {"text": summary.strip(), "tokens": 0}

@app.post("/translate", summary="Translate text", tags=["AI Services"])
def translate_text(request: ProcessRequest):
    if not translator: raise HTTPException(503, "Translation service is currently unavailable.")

    language = request.options.get('language', 'English')
    system_prompt = "You are a helpful assistant that translates text accurately."
    user_prompt = f"Translate the following text to {language}:\n\n{request.text}"
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    translation = translator(prompt, max_new_tokens=1024, temperature=0.2)

    return {"text": translation.strip(), "tokens": 0}
