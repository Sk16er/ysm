# pip install fastapi uvicorn pytubefix whisper nltk
import tempfile
import os
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pytubefix import YouTube
import whisper
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.staticfiles import StaticFiles
# Download NLTK punkt tokenizer
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK punkt: {e}")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Define input model for YouTube URL
class YouTubeURL(BaseModel):
    url: str

def clean_filename(title):
    """Clean video title to make it a valid filename."""
    cleaned = re.sub(r'[\/:*?"<>|]', '_', title)
    cleaned = re.sub(r'[.…]', '_', cleaned)
    return cleaned.strip()

def process_youtube_video(youtube_url):
    """Download and transcribe YouTube video."""
    try:
        youtube = YouTube(youtube_url)
        audio_stream = youtube.streams.filter(only_audio=True).first()
        video_title = clean_filename(youtube.title)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_filename = f"{video_title}.m4a"
            audio_file = os.path.join(tmpdir, audio_filename)
            audio_stream.download(output_path=tmpdir, filename=audio_filename)
            logging.debug(f"Audio downloaded to: {audio_file}")

            if not os.path.exists(audio_file):
                logging.error(f"Audio file {audio_file} not found.")
                return None

            whisper_model = whisper.load_model("base")
            transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()
            return transcription

    except Exception as e:
        logging.error(f"Error processing YouTube video: {e}")
        return None

def format_transcription(transcription):
    """Format transcription into readable sentences."""
    try:
        sentences = sent_tokenize(transcription)
        return "\n".join(sentences)
    except Exception as e:
        logging.error(f"Error formatting transcription: {e}")
        return "\n".join(transcription.split('. '))

@app.post("/transcribe")
async def transcribe_video(data: YouTubeURL):
    """API endpoint to transcribe YouTube video."""
    url = data.url
    if not (url.startswith("https://www.youtube.com/watch?v=") or url.startswith("https://youtu.be/")):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcription = process_youtube_video(url)
    if not transcription:
        raise HTTPException(status_code=500, detail="Failed to process video")

    formatted_transcription = format_transcription(transcription)
    return {"transcription": formatted_transcription}