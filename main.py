
from gradio.routes import mount_gradio_app
from app import app
from ui import demo

app = mount_gradio_app(app, demo, path='/')