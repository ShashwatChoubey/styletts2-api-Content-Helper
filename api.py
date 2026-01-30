from contextlib import asynccontextmanager
import uuid
import logging
from fastapi import FastAPI,BackgroundTasks,Depends,HTTPException,Header
from pydantic import BaseModel
import os
from libri_inference import STYLETTS2_Inference
import re
import numpy as np
import soundfile as sf
from fastapi.security import APIKeyHeader
from huggingface_hub import hf_hub_download
import shutil
import boto3

# from dotenv import load_dotenv
# load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

synthesizer = None
reference_style = None
API_KEY = os.getenv("API_KEY", "your-default-api-key")

api_key_header = APIKeyHeader(name="Authorization",auto_error=False)

async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        logger.warning("No API key provided")
        raise HTTPException(status_code=401,detail="API key is missing")
    
    if authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ","")
    else:
        token = authorization
    
    if token != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401,detail="Invalid API key")
    
    return token


def get_s3_client():
    client_kwargs = {'region_name': os.getenv("AWS_REGION","us-east-1")}
    
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        client_kwargs.update({
            'aws_access_key_id': os.getenv("AWS_ACCESS_KEY_ID"),
            'aws_secret_access_key': os.getenv("AWS_SECRET_ACCESS_KEY")
        })
    
    return boto3.client('s3',**client_kwargs)


def download_models_from_hf():
    """Download model weights from Hugging Face Hub at startup"""
    try:
       
        repo_id = os.getenv("HF_MODEL_REPO", "YOUR_USERNAME/styletts2-models")
        
        os.makedirs("/app/Models", exist_ok=True)
        
        logger.info(f"Downloading models from {repo_id}...")
        
      
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.yml",
            cache_dir="/tmp/hf_cache"
        )
        shutil.copy(config_path, "/app/Models/config.yml")
        logger.info("Config downloaded")
        
        
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="epochs_2nd_00020.pth",
            cache_dir="/tmp/hf_cache"
        )
        shutil.copy(model_path, "/app/Models/epochs_2nd_00020.pth")
        logger.info(" Model weights downloaded")
        
       
        ref_audio_path = hf_hub_download(
            repo_id=repo_id,
            filename="anger.wav",
            cache_dir="/tmp/hf_cache"
        )
        ref_audio_path1 = hf_hub_download(
            repo_id=repo_id,
            filename="narrator.wav",
            cache_dir="/tmp/hf_cache"
        )
        ref_audio_path2 = hf_hub_download(
            repo_id=repo_id,
            filename="women.wav",
            cache_dir="/tmp/hf_cache"
        )
        shutil.copy(ref_audio_path, "/app/Models/anger.wav")
        shutil.copy(ref_audio_path1, "/app/Models/narrator.wav")
        shutil.copy(ref_audio_path2,"/app/Models/woman.wav")
        logger.info(" Reference audio downloaded")
        
        logger.info(" All models downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f" Error downloading models: {e}")
        logger.error("Make sure you've set HF_MODEL_REPO environment variable correctly")
        return False


s3_client = get_s3_client()
S3_PREFIX = os.getenv("S3_PREFIX","styletts2-output")
S3_BUCKET = os.getenv("S3_BUCKET","contenthelper")

@asynccontextmanager
async def lifespan(app:FastAPI):
    global synthesizer,reference_style
    
    
    logger.info("Downloading models from Hugging Face Hub...")
    if not download_models_from_hf():
        logger.error("Failed to download models. API will not work.")
    
    logger.info("Loading StyleTTS2 model ...")
    try:
        synthesizer = STYLETTS2_Inference(
            config_path=os.getenv("CONFIG_PATH","/app/Models/config.yml"),
            model_path=os.getenv("MODEL_PATH","/app/Models/epochs_2nd_00020.pth")
        )
        logger.info("StyleTTS2 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load StyleTTS2 model:{e}")
    
    yield
    
    logger.info("Shutting down StyleTTS2 API")


app = FastAPI(title="StyleTTS2 API",lifespan=lifespan)

Target_Voices= {
    "angry":"/app/Models/anger.wav",
    "narrator":"/app/Models/narrator.wav",
    "women":"/app/Models/women.wav"
}

def text_chunker(text,chunk_size=100):
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        if current_pos + chunk_size >= text_len:
            chunks.append(text[current_pos:])
            break

        chunk_end = current_pos + chunk_size
        search_text = text[current_pos:chunk_end]

        sentence_ends = [m.end() for m in re.finditer(r'[.!?]+',search_text)]

        if sentence_ends:
            last_sentence_end = sentence_ends[-1]
            chunks.append(text[current_pos:current_pos+last_sentence_end])
            current_pos += last_sentence_end
        
        else:
            last_space = search_text.rfind(' ')
            if last_space > 0:
                chunks.append(text[current_pos:current_pos+last_space])
                current_pos += last_space+1
            
            else:
                chunks.append(text[current_pos:chunk_end])
                current_pos += chunk_end
        
        while current_pos < text_len and text[current_pos].isspace():
            current_pos +=1

    return chunks


class TextOnlyRequest(BaseModel):
    text: str
    target_voice: str


@app.post("/generate",dependencies=[Depends(verify_api_key)])
async def generate_speech(request: TextOnlyRequest,background_tasks: BackgroundTasks):
    if len(request.text) > 2500:
        raise HTTPException(
            status_code=400,detail="Text length exceeds the limit of 2500")
    
    if not synthesizer:
        raise HTTPException(status_code=500,detail="Model not loaded")
    
    if request.target_voice not in Target_Voices:
        raise HTTPException(
            status_code=400,detail=f"Target voice not supported.Choose from:{', '.join(Target_Voices.keys())}")
    
    try:
        ref_audio_path = Target_Voices[request.target_voice]

        current_style = synthesizer.compute_style(ref_audio_path)
        logger.info(
            f"Using voice {request.target_voice} from {ref_audio_path}"
        )

        audio_id = str(uuid.uuid4())
        output_filename = f"{audio_id}.wav"
        local_path = f"/tmp/{output_filename}"
        
        text_chunks = text_chunker(request.text)
        logger.info(f"Text split into chunks: {len(text_chunks)}")

        audio_segments = []

        for i,chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)}")

            audio_chunk = synthesizer.inference(
                text = chunk,
                ref_s = current_style
            )

            audio_segments.append(audio_chunk)

            if i < len(text_chunks)-1:
                silence = np.zeros(int(24000*0.3))
                audio_segments.append(silence)

        if len(audio_segments) > 1:
            full_audio = np.concatenate(audio_segments)
        else:
            full_audio = audio_segments[0]
        
        sf.write(local_path,full_audio,24000)

        # Upload to S3
        s3_key = f"{S3_PREFIX}/{output_filename}"
        s3_client.upload_file(local_path,S3_BUCKET,s3_key)

        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params = {'Bucket': S3_BUCKET,'Key': s3_key},
            ExpiresIn = 3600
        )

        background_tasks.add_task(os.remove,local_path)

        return {
            "audio_url":presigned_url,
            "s3_key": s3_key
        }
        
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        raise HTTPException(
            status_code=500,detail=f"Failed to generate speech: {str(e)}"
        )
    
@app.get("/voices",dependencies=[Depends(verify_api_key)])
async def list_voices():
    return {"voices": list(Target_Voices.keys())}

@app.get("/status")
async def status():
    """Health check endpoint - no auth required"""
    if synthesizer:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "not loaded"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "StyleTTS2 API is running",
        "endpoints": {
            "POST /generate": "Generate speech from text (uploads to S3)",
            "GET /voices": "List available voices",
            "GET /status": "Check API health"
        }
    }
