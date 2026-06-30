import subprocess
import os
import requests
import time
import re

# Configuration
API_URL = "http://localhost:11434/generate" # Adjust if your FastAPI port is different
WORKSPACE_DIR = "data/workspace"
CPP_FILE = os.path.join(WORKSPACE_DIR, "solution.cpp")
EXE_FILE = os.path.join(WORKSPACE_DIR, "solution")

def ensure_workspace():
    """Creates a clean workspace directory for compilation."""
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)

def get_cpp_code_from_model(prompt: str) -> str:
    """Fetches the generated C++ code from your local FastAPI server."""
    payload = {
        "prompt": prompt,
        "temperature": 0.1,
        "max_tokens": 1500
    }
    
    # We are hitting the FastAPI layer via the WSL bridge IP
    response = requests.post("http://172.21.16.1:8000/generate", json=payload)
    response.raise_for_status()
    raw_output = response.json().get("code", "")
    
    print("\n--- [DEBUG] RAW LLM OUTPUT ---")
    print(repr(raw_output))
    print("------------------------------\n")
    
    # FIX: Look specifically for cpp/c++ tags first
    ticks = chr(96) * 3
    strict_pattern = rf'{ticks}(?:cpp|c\+\+)\s*\n(.*?){ticks}'
    matches = re.findall(strict_pattern, raw_output, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        # Fallback: grab any generic code blocks, but ONLY keep them if they contain C++ syntax
        loose_pattern = rf'{ticks}\s*\n(.*?){ticks}'
        all_blocks = re.findall(loose_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        matches = [b for b in all_blocks if "#include" in b or "int main" in b or "std::" in b]
    
    if matches:
        # Stitch all extracted C++ blocks together
        extracted_code = "\n\n".join(matches).strip()
    else:
        # Fallback: Extract from first #include to last closing brace
        start_idx = raw_output.find("#include")
        end_idx = raw_output.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted_code = raw_output[start_idx:end_idx+1].strip()
        else:
            extracted_code = raw_output.strip()
            
    return extracted_code

def compile_and_run(cpp_code: str) -> dict:
    """Writes, compiles, and executes the C++ code, capturing all outputs."""
    if not cpp_code.strip():
        return {
            "status": "empty_output_error",
            "error": "The extracted C++ code was completely empty. Check the Raw LLM Output."
        }
        
    ensure_workspace()
    
    # 1. Write the code to disk
    with open(CPP_FILE, "w") as f:
        f.write(cpp_code)
        
    # 2. Compile using g++ (Optimized for C++17)
    compile_cmd = ["g++", "-O3", "-std=c++17", CPP_FILE, "-o", EXE_FILE]
    
    try:
        compile_process = subprocess.run(
            compile_cmd, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        # Check if compilation failed
        if compile_process.returncode != 0:
            return {
                "status": "compilation_failed",
                "error": compile_process.stderr.strip()
            }
            
    except subprocess.TimeoutExpired:
        return {"status": "compilation_timeout", "error": "g++ took too long."}

    # 3. Execute the binary
    try:
        run_process = subprocess.run(
            [f"./{EXE_FILE}"], 
            capture_output=True, 
            text=True, 
            timeout=5 # Hard kill if the model writes an infinite loop
        )
        
        if run_process.returncode != 0:
            return {
                "status": "runtime_error",
                "error": run_process.stderr.strip(),
                "output": run_process.stdout.strip()
            }
            
        return {
            "status": "success",
            "output": run_process.stdout.strip()
        }
        
    except subprocess.TimeoutExpired:
        return {"status": "execution_timeout", "error": "Binary execution timed out (infinite loop?)."}
    finally:
        # Cleanup the executable to prevent stale runs
        if os.path.exists(EXE_FILE):
            os.remove(EXE_FILE)

# --- Test the Pipeline ---
if __name__ == "__main__":
    print("Testing One-Shot Generation and Compilation Pipeline...\n")
    
    test_prompt = "Write a C++17 program that prints the first 10 numbers of the Fibonacci sequence. The main function should return 0. Include <iostream>."
    
    print("1. Requesting code from LLM...")
    start_time = time.time()
    code = get_cpp_code_from_model(test_prompt)
    print(f"Code received in {time.time() - start_time:.2f} seconds.")
    
    print("\n--- EXTRACTED C++ CODE TO COMPILE ---")
    print(code)
    print("-------------------------------------\n")
    
    print("2. Compiling and Executing...")
    result = compile_and_run(code)
    
    print("\n--- PIPELINE RESULT ---")
    if result["status"] == "success":
        print("[PASS] Execution Output:")
        print(result["output"])
    else:
        print(f"[FAIL] {result['status'].upper()}")
        print(f"Error Details:\n{result.get('error', '')}")
