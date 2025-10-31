import re
import json
import time
import requests
from pathlib import Path
from threading import Thread
from requests.adapters import HTTPAdapter
from config_general import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_FACTOR, HTTP_POOL_CONNECTIONS, HTTP_POOL_MAXSIZE

_session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=HTTP_POOL_CONNECTIONS,
    pool_maxsize=HTTP_POOL_MAXSIZE,
    max_retries=0
)
_session.mount('https://', adapter)
_session.mount('http://', adapter)

class RequestWithTimeout:
    def __init__(self):
        self.result = None
        self.exception = None
    
    def make_request(self, url, headers, json_data):
        try:
            self.result = _session.post(url, headers=headers, json=json_data, timeout=30)
        except Exception as e:
            self.exception = e

def _make_openrouter_request(prompt, model_name, api_key, temperature=0.0, max_tokens=None, reasoning_effort=None, reasoning_max_tokens=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    if reasoning_effort or reasoning_max_tokens:
        reasoning_config = {}
        if reasoning_effort:
            reasoning_config["effort"] = reasoning_effort
        if reasoning_max_tokens:
            reasoning_config["max_tokens"] = reasoning_max_tokens
        data["reasoning"] = reasoning_config
    
    req = RequestWithTimeout()
    thread = Thread(target=req.make_request, args=(url, headers, data))
    thread.daemon = True
    thread.start()
    thread.join(timeout=REQUEST_TIMEOUT)
    
    if thread.is_alive():
        raise requests.Timeout(f"Request exceeded {REQUEST_TIMEOUT} second timeout")
    
    if req.exception:
        raise req.exception
    
    if req.result is None:
        raise Exception("Request returned no result")
    
    return req.result.json()

def call_openrouter(prompt, model_name, api_key, temperature=0.0, reasoning_effort=None, reasoning_max_tokens=None, max_tokens=None, run_id=None, record_id=None, context=None):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response_json = _make_openrouter_request(prompt, model_name, api_key, temperature, max_tokens, reasoning_effort, reasoning_max_tokens)
            
            if 'choices' in response_json and len(response_json['choices']) > 0:
                message = response_json['choices'][0]['message']
                return {
                    'content': message.get('content'),
                    'reasoning': message.get('reasoning'),
                    'reasoning_details': message.get('reasoning_details')
                }, response_json.get('usage', {})
            
            error_text = response_json.get('error', {}).get('message', 'Unknown error') if 'error' in response_json else 'No response from model'
            raise Exception(f"API Error: {error_text}")
            
        except requests.Timeout:
            if attempt < MAX_RETRIES:
                print(f"Timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}) - Run: {run_id} - {context or 'Request'}")
                time.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            else:
                raise Exception(f"Request timeout after {MAX_RETRIES + 1} attempts")
        except Exception as e:
            if "API Error:" in str(e) or "Request timeout" in str(e):
                raise
            raise Exception(str(e))

def log_progress(status_type, count, total, run_id, record_id, api_key, start_usage, error=None, is_correct=None):
    key_info = get_openrouter_key_info(api_key)
    current_usage = key_info.get('data', {}).get('usage', 0)
    cost = current_usage - start_usage
    
    if status_type == "completed":
        extra = f"Correct: {is_correct}" if is_correct is not None else ""
        print(f"Completed {count}/{total} - Run ID: {run_id} - Record ID: {record_id} - {extra} - Cost: ${cost:.6f}")
    else:
        error_msg = str(error)[:100] if error else "Unknown error"
        print(f"Failed {count}/{total} - Run ID: {run_id} - Record ID: {record_id} - Error: {error_msg} - Cost: ${cost:.6f}")
    
    return cost

def get_openrouter_key_info(api_key):
    if not api_key:
        return None
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Warning] Could not fetch OpenRouter key info: {e}")
    return None

def parse_answer(response_text):
    parsed = {
        'is_valid': False,
        'answer': None,
        'confidence': None,
        'reasoning': None
    }
    
    final_answer_match = re.search(r'<BEGIN FINAL ANSWER>(.*?)</?END FINAL ANSWER>', response_text, re.DOTALL | re.IGNORECASE)
    if not final_answer_match:
        return parsed
    
    final_answer_text = final_answer_match.group(1)
    parsed['is_valid'] = True
    
    answer_match = re.search(r'Answer:\s*(\d+)', final_answer_text, re.IGNORECASE)
    if answer_match:
        parsed['answer'] = int(answer_match.group(1))
    
    confidence_match = re.search(r'Confidence:\s*(\d+)(?:\.\d+)?%?', final_answer_text, re.IGNORECASE)
    if confidence_match:
        parsed['confidence'] = int(confidence_match.group(1))
    
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\s*$|\Z)', final_answer_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        parsed['reasoning'] = reasoning_match.group(1).strip()
    
    return parsed

def parse_debater_response(response_text, private_scratchpad):
    public_match = re.search(r'<BEGIN PUBLIC ARGUMENT>(.*?)</?END PUBLIC ARGUMENT>', response_text, re.DOTALL | re.IGNORECASE)
    
    if not public_match:
        return None, "Missing <BEGIN PUBLIC ARGUMENT> or </END PUBLIC ARGUMENT> tags"
    
    parsed = {'public_argument': public_match.group(1).strip()}
    
    if private_scratchpad:
        private_match = re.search(r'<BEGIN PRIVATE REASONING>(.*?)</?END PRIVATE REASONING>', response_text, re.DOTALL | re.IGNORECASE)
        if not private_match:
            return None, "Missing <BEGIN PRIVATE REASONING> or </END PRIVATE REASONING> tags"
        parsed['private_reasoning'] = private_match.group(1).strip()
    
    return parsed, None
