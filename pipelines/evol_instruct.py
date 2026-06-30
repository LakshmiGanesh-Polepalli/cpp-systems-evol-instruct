import os
import json
import time
import requests
from datasets import load_dataset

print("Loading dataset footprint...")
dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response", split="train")

def is_cpp_optimization(example):
    text = (example["instruction"] + " " + example["response"]).lower()
    return any(k in text for k in ["c++", "cpp", "std::", "pointers"]) and \
           any(k in text for k in ["time complexity", "memory", "optimize", "o(n)"])

sample_seeds = dataset.filter(is_cpp_optimization).select(range(500))

# --- EXPLICIT ABSOLUTE PATHS ---
# This guarantees the file always saves in the exact same place!
output_file = "/home/admin_/ml_project/cpp_systems_qlora.jsonl"
failed_file = "/home/admin_/ml_project/failed_seeds.jsonl"

# Ensure the main project directory exists
os.makedirs("/home/admin_/ml_project", exist_ok=True)

# Count existing rows to resume seamlessly
existing_count = sum(1 for _ in open(output_file)) if os.path.exists(output_file) else 0
print(f"Found {existing_count} rows. Resuming using local Ollama...")

def generate_robust_evolution(instruction, max_retries=3):
    """Attempts to evolve a prompt with built-in retries and JSON enforcement."""
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""You are an expert C++ Systems Engineer. 
    Take this base coding instruction: "{instruction}"
    1. EVOLVE it: Rewrite this instruction to make it significantly harder. Require advanced C++ features.
    2. SOLVE it: Provide the optimal, highly-commented C++ solution.
    Output EXACTLY as a valid JSON object with keys: "new_instruction" and "new_response"."""

    for attempt in range(max_retries):
        try:
            payload = {
                "model": "qwen2.5-coder:7b",
                "prompt": prompt,
                "format": "json",  # Forces Ollama to output valid JSON
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
            
            # Dynamic Timeouts: Gives the model more time on retry attempts
            current_timeout = 150 + (attempt * 30) 
            
            response = requests.post(url, json=payload, timeout=current_timeout)
            response.raise_for_status()
            
            raw_text = response.json().get("response", "")
            data = json.loads(raw_text)
            
            # Strict Schema Validation
            if "new_response" not in data or "new_instruction" not in data:
                raise ValueError("Missing required keys in JSON")
                
            return data 
            
        except requests.exceptions.Timeout:
            print(f"  [Attempt {attempt+1}/{max_retries}] Timeout. Retrying...")
            time.sleep(2) # Brief pause to let local GPU clear memory
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] JSON Error: {e}. Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] Unexpected Error: {e}. Retrying...")
            time.sleep(1)
            
    # If it fails all 3 retries, return None
    return None

# --- MAIN LOOP ---
# Open both files in append mode
with open(output_file, "a") as f, open(failed_file, "a") as err_f:
    for i in range(existing_count, len(sample_seeds)):
        row = sample_seeds[i]
        
        # Pass the instruction to our robust function
        evolved = generate_robust_evolution(row['instruction'])
        
        if evolved:
            # Successfully generated and validated JSON! Format it for ShareGPT.
            sharegpt_row = {
                "conversations": [
                    {"from": "human", "value": evolved["new_instruction"]},
                    {"from": "gpt", "value": evolved["new_response"]}
                ]
            }
            f.write(json.dumps(sharegpt_row) + "\n")
            f.flush()
            print(f"Successfully evolved row {i+1}/{len(sample_seeds)}")
        else:
            # Failed all 3 retries. Save to dead-letter queue so we don't lose the base seed.
            print(f"FAILED row {i+1} after 3 retries. Saving to DLQ.")
            err_f.write(json.dumps({"failed_base_instruction": row['instruction']}) + "\n")
            err_f.flush()

print(f"Run complete. Check {output_file} for your final dataset.")
