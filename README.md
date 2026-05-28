# Hate Speech Verification Using LLM

This repository contains the code for a Language Model (LLM) Stand, built with FastAPI and Gradio, designed for analyzing text to determine the presence of hate speech.

## Overview
The project utilizes advanced language models to provide an automated and efficient solution for hate speech detection. It is intended for research, content moderation, and other applications promoting safe and inclusive online communication.

### Key Features
- **FastAPI Backend**: For a robust and performant server-side application.
- **Gradio Frontend**: For building user-friendly interfaces to interact with the hate speech analysis model.
- **Hate Speech Detection**: Analyzing text input to determine if it contains hate speech.

## Installation
To get started with the project, follow these steps:

### Clone the Repository
```bash
git clone https://github.com/kaengreg/ispras_hate_verification.git
cd ispras_hate_verification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Make sure you have Python 3.11+ installed on your system.

### Optional vLLM model endpoints

By default, both tabs use `VLLM_BASE_URL` and `VLLM_API_KEY`. `VLLM_BASE_URL` may expose several models through `/v1/models`.

To add models from one more vLLM endpoint, set `ADDITIONAL_VLLM_BASE_URL`. The app will call `/v1/models` on both endpoints, merge all model ids into the dropdowns, and route `/run` or `/generate` to the endpoint where the selected model is available.

Example:

```bash
VLLM_BASE_URL="http://lab-host:6266"
VLLM_API_KEY="..."

ADDITIONAL_VLLM_BASE_URL="http://my-host:8000"
ADDITIONAL_VLLM_API_KEY=""
```

The `/models` endpoint returns the combined list from `VLLM_BASE_URL` and `ADDITIONAL_VLLM_BASE_URL`. Both moderation and generation dropdowns use this same list. If the same model id exists on both endpoints, the UI shows source-specific choices for that duplicate. Otherwise the model is shown once by its regular model id.

For the comparison tab, configure candidate model names with comma-separated lists:

```bash
BASE_COMPARISON_MODELS="RuadaptQwen3-4B-Instruct,another-base-model-name"
FINE_TUNED_COMPARISON_MODELS="RuadaptQwen3-4B-Instruct_sft_pro,RuadaptQwen3-4B-Instruct_sft_stance,RuadaptQwen3-4B_simpo_stance_v4"
```

The baseline is selected as the first available model from `BASE_COMPARISON_MODELS` in the combined `/models` list, so it may live on either `VLLM_BASE_URL` or `ADDITIONAL_VLLM_BASE_URL`.

## Usage
Run the FastAPI and Gradio-based LLM Stand using the following command:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000  
```
Alternatively, launch it using Docker (requires VLLM_BASE_URL and VLLM_API_KEY to be set in the .env file):
```bash
./run_stand.sh
```

This will start the FastAPI server and launch the Gradio interface for user interaction.

### Access the Interface
Once the server is running, you can:
- Open the Gradio interface at `http://localhost:8000` to analyze the text.
- Access the FastAPI docs at: `http://localhost:8000/docs`

## License
This project is licensed under the [MIT License](LICENSE).
