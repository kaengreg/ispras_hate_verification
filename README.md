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

Make sure you have Python 3.8+ installed on your system.

## Usage
Run the FastAPI and Gradio-based LLM Stand using the following command:
```bash
python main.py
```
This will start the FastAPI server and launch the Gradio interface for user interaction.

### Access the Interface
Once the server is running, you can:
- Access the FastAPI docs at: `http://127.0.0.1:8000/docs`
- Open the Gradio interface to analyze the text.

## Contributing
Contributions are welcome! Feel free to create issues or submit pull requests for any feature requests, bug fixes, or enhancements.

## License
This project is licensed under the [MIT License](LICENSE).