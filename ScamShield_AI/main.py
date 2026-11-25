import os
import gradio as gr
from fastapi import FastAPI
from gradio.routes import mount_gradio_app

# Import the existing scam detection logic
# Assuming check_rules and related constants are defined in app.py
from app import check_rules

def analyze_sms(text: str) -> str:
    """Analyze an SMS message and return a human‑readable summary."""
    hits = check_rules(text)
    if not hits:
        return "✅ No scam patterns detected."
    lines = []
    for h in hits:
        severity = "⚠️ Critical" if h.get("critical") else "⚠️ Warning"
        lines.append(f"{severity}: {h.get('name')} – {h.get('why')}")
    return "\n".join(lines)

# Gradio interface definition (same as original app)

demo = gr.Interface(
    fn=analyze_sms,
    inputs=gr.Textbox(lines=4, placeholder="Paste an SMS here…"),
    outputs=gr.Markdown(),
    title="ScamShield AI",
    description="Detect common phishing/scam patterns in Indian SMS messages.",
    article="Built with Gradio & FastAPI for Vercel deployment."
)

# Wrap the Gradio demo in a FastAPI app for Vercel
app = FastAPI()
app = mount_gradio_app(app, demo, path="/")
