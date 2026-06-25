from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="C++ Systems Assistant API")

# Define the structured payload
class CodeRequest(BaseModel):
    prompt: str
    temperature: float = 0.1
    max_tokens: int = 1024
    
@app.post("/generate")
async def generate_cpp_code(request: CodeRequest):
    """Relaxed stop tokens to prevent model collapse, sanitized at the interface."""
    try:
        payload = {
            "model": "qwen2.5-cpp-custom",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a ruthless, highly optimized C++ systems engineer. Output strictly standard C++17 code. No yapping. No explanations."
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                # Stop only at logical code boundaries, not control tokens
                "stop": ["```\n", "You are Qwen"] 
            }
        }
        
        import requests
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()
        
        result = response.json()
        raw_code = result["message"]["content"]
        
        # Sanitize identity strings
        cleaned_code = raw_code.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.:semicolon", "")
        
        return {"status": "success", "code": cleaned_code.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Spins up the server on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
