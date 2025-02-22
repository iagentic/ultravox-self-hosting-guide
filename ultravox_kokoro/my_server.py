import asyncio
import base64
import json
import os
import time
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urljoin

import numpy as np
import soundfile as sf
import transformers
from fastapi import FastAPI, WebSocket, Header, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from kokoro import KPipeline
from pydantic import BaseModel
import torch
from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Voice Configuration
VOICES = {
    "Bella (US Female)": {"code": "af_bella", "lang_code": "a"},
    "Nicole (US Female)": {"code": "af_nicole", "lang_code": "a"},
    "Michael (US Male)": {"code": "am_michael", "lang_code": "a"},
    "Emma (UK Female)": {"code": "bf_emma", "lang_code": "b"},
    "George (UK Male)": {"code": "bm_george", "lang_code": "b"}
}

# Voice Activity Detection (VAD) Class
class SpeechDetector:
    def __init__(self):
        logger.info("Loading Silero VAD model...")
        self.model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        self.model.eval()
        self.sample_rate = 16000
        self.window_size = 1536
        self.threshold = 0.5
        self.audio_window = deque(maxlen=10)
        self.speech_probs = deque(maxlen=10)

    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        try:
            tensor = torch.FloatTensor(audio_chunk).unsqueeze(0)
            prob = self.model(tensor, self.sample_rate).item()
            self.speech_probs.append(prob)
            return np.mean(self.speech_probs) > self.threshold
        except Exception as e:
            logger.error(f"VAD Error: {e}")
            return False

# Text-to-Speech Class
class TextToSpeech:
    def __init__(self):
        logger.info("Preloading TTS models...")
        self.tts_pipelines = {voice: KPipeline(lang_code=config["lang_code"]) for voice, config in VOICES.items()}

    def synthesize(self, voice_name: str, text: str) -> np.ndarray:
        voice_config = VOICES[voice_name]
        pipeline = self.tts_pipelines[voice_name]
        audio_segments = []
        for _, _, audio_data in pipeline(text, voice=voice_config["code"], speed=1):
            audio_segments.append(audio_data)
        return np.concatenate(audio_segments) if audio_segments else np.array([])

# AI Response Generator Class
class AIResponseGenerator:
    def __init__(self):
        logger.info("Loading Ultravox Model...")
        self.pipeline = transformers.pipeline(model='fixie-ai/ultravox-v0_4', trust_remote_code=True)

    def generate_response(self, system_prompt: str, audio: np.ndarray) -> str:
        result = self.pipeline({'audio': audio, 'turns': [{'role': 'system', 'content': system_prompt}], 'sampling_rate': 16000}, max_new_tokens=200)
        return result[0] if isinstance(result, list) else str(result)

# Request Body Schema
class CallConfig(BaseModel):
    systemPrompt: str
    temperature: float = 0.8
    voice: Optional[str] = None

# FastAPI App Initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models at server startup
speech_detector = SpeechDetector()
tts = TextToSpeech()
ai_response_generator = AIResponseGenerator()

def validate_api_key(x_api_key: Optional[str]):
    required_key = os.getenv("ULTRAVOX_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/calls")
@limiter.limit("5/minute")
async def create_call(config: CallConfig, request: Request, x_api_key: Optional[str] = Header(None)):
    validate_api_key(x_api_key)

    call_id = f"call_{int(time.time())}"
    base_url = f"ws://{request.headers.get('host')}/api/calls/{call_id}/join"
    return {"callId": call_id, "joinUrl": base_url, "status": "success", "config": config.dict()}

@app.websocket("/api/calls/{call_id}/join")
async def join_call(websocket: WebSocket, call_id: str):
    await websocket.accept()
    logger.info(f"WebSocket connected: {call_id}")
    
    try:
        await websocket.send_json({"type": "state", "state": "speaking"})
        await websocket.send_json({"type": "transcript", "role": "agent", "text": "Hello! How can I assist you today?", "final": True})
        
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            if "bytes" in message:
                audio_data = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                if speech_detector.detect_speech(audio_data):
                    response_text = ai_response_generator.generate_response("You are a helpful assistant.", audio_data)
                    audio_response = tts.synthesize("Bella (US Female)", response_text)
                    audio_bytes = (audio_response * 32768).astype(np.int16).tobytes()
                    await websocket.send_bytes(audio_bytes)
    except asyncio.TimeoutError:
        logger.warning(f"WebSocket timeout for call {call_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket for call {call_id}: {e}")
    finally:
        await websocket.close()
        logger.info(f"WebSocket connection closed: {call_id}")

# Server Run
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Speech Server...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
