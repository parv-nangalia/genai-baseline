import os
import requests
import time

def process_gateway_request(payload: dict) -> dict:
    target = os.getenv("GATEWAY_TARGET", "ollama").lower()
    
    if target == "litellm":
        litellm_url = f"{os.getenv('LITELLM_URL', 'http://localhost:4000').rstrip('/')}/v1/chat/completions"
        try:
            response = requests.post(litellm_url, json=payload, timeout=120.0)
            if response.status_code != 200:
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text
                raise ValueError(f"LiteLLM server error (HTTP {response.status_code}): {error_detail}")
            return response.json()
        except Exception as e:
            raise ValueError(f"LiteLLM gateway forwarding error: {str(e)}")
            
    elif target == "ollama":
        ollama_url = f"{os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')}/api/chat"
        model = payload.get("model", os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct"))
        messages = payload.get("messages", [])
        
        ollama_payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(ollama_url, json=ollama_payload, timeout=120.0)
            if response.status_code != 200:
                try:
                    error_detail = response.json().get("error", response.text)
                except Exception:
                    error_detail = response.text
                raise ValueError(f"Ollama server error (HTTP {response.status_code}): {error_detail}")
            
            result = response.json()
            message_content = result.get("message", {}).get("content", "").strip()
            
            return {
                "id": f"chatcmpl-ollama-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": message_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages).split()),
                    "completion_tokens": len(message_content.split()),
                    "total_tokens": len(str(messages).split()) + len(message_content.split())
                }
            }
        except Exception as e:
            raise ValueError(f"Ollama gateway forwarding error: {str(e)}")
    else:
        raise ValueError(f"Unsupported GATEWAY_TARGET: '{target}'")
