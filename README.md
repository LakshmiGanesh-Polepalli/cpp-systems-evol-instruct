# ⚙️ Cpp-Systems-Evol-Instruct

An end-to-end machine learning pipeline that synthesizes highly complex C++ programming data locally and fine-tunes a quantized LLM (Qwen2.5-Coder-7B) using QLoRA. The goal is to distill advanced systems architecture reasoning into a highly efficient, locally deployable model.

# 🚀 Architecture Overview

This project is divided into two distinct MLOps phases, utilizing a hybrid Local/Cloud compute strategy:

Phase 1: Data Engineering (Local WSL2)

Utilizes an auto-resuming API mutation loop via a local Qwen2.5-coder:7b Ollama instance.

Takes standard C++ coding instructions and "evolves" them to require advanced systems-level constraints (concurrency, memory management, algorithmic complexity).

Resilience: Implements strict JSON schema enforcement and network timeout severance to automatically filter out LLM hallucinations and infinite generation loops.

Phase 2: Model Fine-Tuning (Google Colab T4)

Uses Unsloth to load the base model in 4-bit precision, drastically reducing VRAM overhead.

Trains custom LoRA adapters (Rank 16) on the synthesized dataset using Supervised Fine-Tuning (SFT).

# 📊 Pipeline Telemetry & Yield

Because LLMs hallucinate during synthetic data generation, the Evol-Instruct pipeline actively monitors and drops corrupted outputs. Below are the final yields from the local generation run:

Target Seeds: 500

Successful Evolutions: [X]

Filtered (Timeouts/Infinite Loops): [X]

Filtered (JSON Schema Fails): [X]

Final Dataset Yield: [X]%

# 📂 Project Structure

cpp-systems-evol-instruct/
├── pipelines/
│   └── evol_instruct.py        # The local data generation and mutation script
├── training/
│   └── qlora_unsloth.ipynb     # The Unsloth training notebook for Google Colab
├── data/
│   └── processed/              # Contains the final cpp_systems_qlora.jsonl dataset
├── .gitignore
├── requirements.txt            
└── README.md


# 🛠️ How to Run

1. Generating the Synthetic Data (Local)

Ensure you have Ollama installed and the base model downloaded:

ollama run qwen2.5-coder:7b


Install the required Python dependencies:

pip install -r requirements.txt


Run the generation pipeline. Note: This script features auto-resume. If interrupted, it will safely pick up exactly where it left off.

python pipelines/evol_instruct.py


2. Fine-Tuning the Model (Cloud)

Upload the generated cpp_systems_qlora.jsonl file to your Google Drive.

Open training/qlora_unsloth.ipynb in Google Colab.

Enable the T4 GPU (Runtime -> Change runtime type).

Run all cells to initialize Unsloth, attach the LoRA adapters, and begin training.
