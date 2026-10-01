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

Model routing reuses `/v1/models` responses for 30 seconds by default (`MODEL_ROUTE_CACHE_TTL`; set to `0` to disable). The `/models` endpoint still refreshes the list. Each `/run` or `/generate` request reuses one HTTP client for model lookup and completions.

The `/run` response uses `pass`, `violation`, or `UNDEFINED` as its normalized verdict. For multiclass criteria, `raw_verdict` also contains the original class (for example, `NEU` or `POS`).

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

The uvicorn and Docker commands above start the FastAPI server and launch the Gradio interface for user interaction.

### SLAVA benchmark

The evaluator uses the benchmark's official `DataLoader`, `OpenAIModel`, `ModelHandler`,
`ModelEval.run_evaluation()` and `MetricsCalculator` pipeline from
[`ikanam-ai/SLAVA`](https://github.com/ikanam-ai/SLAVA). Install the updated dependencies and run a model served through an OpenAI-compatible API (for example, vLLM):

```bash
pip install -r requirements.txt

python model_testing/slava_bench.py \
  --model RuadaptQwen3-4B-Instruct \
  --api-base http://127.0.0.1:8000 \
  --results-dir results/slava
```

The official loader downloads `RANEPA-ai/SLAVA-OpenData-2800-v1` on the first run. Use `--limit 20` for a smoke test. The pipeline writes the official predictions CSV and an Excel workbook with the official metrics: EM/LR/F1 for open questions and EM/substring/partial match for non-open questions, grouped by subject, question type, and provocativeness. Use `--skip-metrics` if only model responses are needed.

### Access the Interface
Once the server is running, you can:
- Open the Gradio interface at `http://localhost:8000` to analyze the text.
- Access the FastAPI docs at: `http://localhost:8000/docs`

### Moderation performance benchmark

The batch moderation script records end-to-end latency and throughput. Use a warm-up before the measured requests and describe the hardware in the report:

```bash
python criterion_evaluation.py \
  --input data/guard_train/splits/StanceRus_24_08_test.jsonl \
  --output benchmark/results.jsonl \
  --model RuadaptQwen3-32B_StanceRus_r16_a32_lr2e-4_tbs32_ep2_multitask \
  --criteria anti_russia \
  --concurrency 8 \
  --warmup 3 \
  --hardware "1x NVIDIA H100 80GB"
```

The summary is printed after the run and saved to `benchmark/results.jsonl.stats.json`. It contains p50/p90/p95/p99 latency, successful texts per second, criterion evaluations per second, failures, input length statistics, concurrency and benchmark configuration. Omit `--criteria` to benchmark the complete moderation pipeline. Use `--include-timing` only when per-row latency is also needed in the result JSONL.

## License
This project is licensed under the [MIT License](LICENSE).
