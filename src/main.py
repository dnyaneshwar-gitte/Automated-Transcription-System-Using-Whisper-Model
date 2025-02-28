import os
import time
import ctypes
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import whisper
import json
import warnings
import subprocess


os.environ['LIBC'] = 'msvcrt.dll'
ctypes.CDLL('msvcrt.dll')



DATA_DIR = './data'
TRANSCRIPTIONS_DIR = './data/transcriptions'
PROCESSED_FILES_LOG = 'processed_files.json'
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.aac', '.m4a']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.mov', '.flv']
TEMP_AUDIO_FILE = 'temp_audio.wav'


os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)


def load_processed_files(log_path):
    try:
        with open(log_path, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_processed_files(log_path, processed_files):
    with open(log_path, 'w') as f:
        json.dump(list(processed_files), f)

def extract_audio_from_video(video_path, output_audio_path=TEMP_AUDIO_FILE):
    try:
        command = [
            'ffmpeg', '-y', '-i', video_path, '-vn',
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ]
        subprocess.run(command, check=True)
        return output_audio_path
    except Exception as e:
        print(f'Error extracting audio from {video_path}: {e}')
        return None

def transcribe_file(file_path, model, processed_files):
    if file_path in processed_files:
        return

    if file_path.lower().endswith('.txt'):
        print(f'Skipping .txt file: {file_path}')
        return

    try:
        if any(file_path.lower().endswith(ext) for ext in SUPPORTED_AUDIO_FORMATS):
            audio_path = file_path
        elif any(file_path.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS):
            audio_path = extract_audio_from_video(file_path)
            if not audio_path:
                return
        else:
            print(f'Unsupported file format: {file_path}')
            return

        print(f'Transcribing: {file_path}')
        result = model.transcribe(audio_path, language='hi')  # Set language to Hindi
        transcript_path = os.path.join(TRANSCRIPTIONS_DIR, os.path.basename(file_path) + '.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        print(f'Transcription saved to: {transcript_path}')
        processed_files.add(file_path)
        save_processed_files(PROCESSED_FILES_LOG, processed_files)

        if audio_path == TEMP_AUDIO_FILE and os.path.exists(audio_path):
            os.remove(audio_path)

    except Exception as e:
        print(f'Error transcribing {file_path}: {e}')


class FileHandler(FileSystemEventHandler):
    def __init__(self, model, processed_files):
        self.model = model
        self.processed_files = processed_files

    def on_created(self, event):
        if not event.is_directory:
            transcribe_file(event.src_path, self.model, self.processed_files)

def monitor_directory(directory, model, processed_files):
    event_handler = FileHandler(model, processed_files)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    model = whisper.load_model('large')
    processed_files = load_processed_files(PROCESSED_FILES_LOG)

   
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            transcribe_file(file_path, model, processed_files)

    monitor_directory(DATA_DIR, model, processed_files)
