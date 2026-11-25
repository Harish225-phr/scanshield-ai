import multiprocessing
import os
import sys
import threading

# Vercel / AWS Lambda lacks /dev/shm, causing multiprocessing.Lock() (used by
# Gradio's CSVLogger in gr.Examples) to fail with FileNotFoundError.
multiprocessing.Lock = threading.Lock

# Ensure the ScamShield_AI directory is on sys.path when invoked from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from app import demo

# Wrap the full Gradio demo from app.py in a FastAPI app for Vercel
app = FastAPI()
app = mount_gradio_app(app, demo, path="/")
