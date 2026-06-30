import os
import json
import time
import requests
import subprocess
import re
from datasets import load_dataset

# --- CONFIGURATION ---
OLLAMA_URL = "http://172.21.16.1:11434/api/generate" # Hitting Windows Ollama directly from WSL
WORKSPACE_DIR = "data/workspace"
CPP_FILE = os.path.join(WORKSPACE_DIR, "solution.cpp")
EXE_FILE = os.path.join(WORKSPACE_DIR, "solution")

# Output files
output_file = "/home/admin_/ml_project/cpp_systems_qlora.jsonl"
failed_file = "/home/admin_/ml_project/failed_seeds.jsonl"
os.makedirs("/home/admin_/ml_project", exist_ok=True)

# --- WORKSPACE & COMPILER FUNCTIONS ---
def ensure_workspace():
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

def extract_cpp_code(raw_text: str) -> str:
    """Extracts C++ code from LLM response safely."""
    ticks = chr(96) * 3
    pattern = rf'{ticks}(?:cpp|c\+\+)?\n?(.*?){ticks}'
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return "\n\n".join(matches).strip()
        
    # Fallback
    start_idx = raw_text.find("#include")
    end_idx = raw_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return raw_text[start_idx:end_idx+1].strip()
        
    return raw_text.strip()

def compile_and_run(cpp_code: str) -> dict:
    """Compiles and executes the C++ code to verify it actually works."""
    if not cpp_code.strip():
        return {"status": "empty_code", "error": "No C++ code found."}
        
    ensure_workspace()
    
    with open(CPP_FILE, "w") as f:
        f.write(cpp_code)
        
    # Compile
    compile_cmd = ["g++", "-O3", "-std=c++17", CPP_FILE, "-o", EXE_FILE]
    try:
        compile_process = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=10)
        if compile_process.returncode != 0:
            return {"status": "compilation_failed", "error": compile_process.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"status": "compilation_timeout", "error": "g++ took too long."}

    # Execute
    try:
        run_process = subprocess.run([f"./{EXE_FILE}"], capture_output=True, text=True, timeout=5)
        if run_process.returncode != 0:
            return {"status": "runtime_error", "error": run_process.stderr.strip()}
        return {"status": "success", "output": run_process.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"status": "execution_timeout", "error": "Binary execution timed out."}
    finally:
        if os.path.exists(EXE_FILE):
            os.remove(EXE_FILE)

# --- GENERATION FUNCTION ---
def generate_and_validate(instruction, max_retries=3):
    """Generates code and explicitly verifies it using the compiler."""
    prompt = f"""You are an expert C++ Systems Engineer. 
    Take this base coding instruction: "{instruction}"
    1. EVOLVE it: Rewrite this instruction to make it significantly harder. Require advanced C++ features.
    2. SOLVE it: Provide the optimal, highly-commented C++ solution. It MUST include a main() function that returns 0.
    Output EXACTLY as a valid JSON object with keys: "new_instruction" and "new_response"."""

    for attempt in range(max_retries):
        try:
            payload = {
                "model": "qwen2.5-coder:7b", # Using standard ollama model for speed
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9}
            }
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=150 + (attempt * 30))
            response.raise_for_status()
            
            data = response.json()
            llm_json = json.loads(data.get("response", "{}"))
            
            if "new_response" not in llm_json or "new_instruction" not in llm_json:
                raise ValueError("Missing JSON keys")
            
            # Extract and Validate Code
            cpp_code = extract_cpp_code(llm_json["new_response"])
            validation = compile_and_run(cpp_code)
            
            if validation["status"] == "success":
                return llm_json # It compiled and ran perfectly!
            else:
                print(f"  [Attempt {attempt+1}] Code failed to compile/run: {validation['status']}. Retrying...")
                time.sleep(1)
                
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Error: {e}. Retrying...")
            time.sleep(2)
            
    return None

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("Loading dataset footprint...")
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response", split="train")

    def is_cpp_optimization(example):
        text = (example["instruction"] + " " + example["response"]).lower()
        return any(k in text for k in ["c++", "cpp", "std::", "pointers"])

    sample_seeds = dataset.filter(is_cpp_optimization).select(range(500))
    existing_count = sum(1 for _ in open(output_file)) if os.path.exists(output_file) else 0
    print(f"Found {existing_count} rows. Starting verified generation loop...")

    with open(output_file, "a") as f, open(failed_file, "a") as err_f:
        for i in range(existing_count, len(sample_seeds)):
            row = sample_seeds[i]
            print(f"\nProcessing seed {i+1}/{len(sample_seeds)}...")
            
            evolved = generate_and_validate(row['instruction'])
            
            if evolved:
                sharegpt_row = {
                    "conversations": [
                        {"from": "human", "value": evolved["new_instruction"]},
                        {"from": "gpt", "value": evolved["new_response"]}
                    ]
                }
                f.write(json.dumps(sharegpt_row) + "\n")
                f.flush()
                print(f"  -> [SUCCESS] Compiled successfully. Saved to dataset.")
            else:
                print(f"  -> [FAILED] Could not generate compiling code after 3 retries. Saved to DLQ.")
                err_f.write(json.dumps({"failed_base_instruction": row['instruction']}) + "\n")
                err_f.flush()
