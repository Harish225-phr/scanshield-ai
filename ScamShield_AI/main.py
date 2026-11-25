import os
import sys

# Ensure the ScamShield_AI directory is on sys.path when invoked from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from gradio.routes import mount_gradio_app

from app import demo

# Wrap the full Gradio demo from app.py in a FastAPI app for Vercel
app = FastAPI()
app = mount_gradio_app(app, demo, path="/")
