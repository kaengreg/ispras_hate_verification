import gradio as gr

from app import app
from ui import demo


APP_CSS = """
.gradio-container {
    --color-accent: #1f6feb;
    --slider-color: #1f6feb;
}

.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
}

.brand-logo {
    width: 36px;
    height: 36px;
    object-fit: contain;
}

.brand-logo-fallback {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    background: #1f6feb;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    letter-spacing: 0;
}

.brand-title {
    font-size: 22px;
    font-weight: 700;
}

.gradio-container button[role="tab"][aria-selected="true"] {
    color: #1f6feb !important;
    border-color: #1f6feb !important;
}

.gradio-container button[role="tab"][aria-selected="true"]::after {
    background: #1f6feb !important;
}

.gradio-container input[type="range"] {
    accent-color: #1f6feb;
}
"""


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    css=APP_CSS,
    theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue")
    )
