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

def log_llm_error(run_id, record_id, error_message, error_log_dir='results/debates/error_logs'):
    error_log_path = Path(error_log_dir)
    error_log_path.mkdir(parents=True, exist_ok=True)
    error_file = error_log_path / f'{run_id}.txt'
    
    with open(error_file, 'a') as f:
        f.write('=' * 80 + '\n')
        f.write(f'Run ID: {run_id}\n')
        f.write(f'Record ID: {record_id}\n')
        f.write('=' * 80 + '\n')
        f.write(f'{error_message}\n')
        f.write('=' * 80 + '\n\n')

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

def call_openrouter(prompt, model_name, api_key, temperature=0.0, reasoning_effort=None, reasoning_max_tokens=None, max_tokens=None, run_id=None, record_id=None, context=None, error_log_dir='results/debates/error_logs'):
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
            
            error_msg = json.dumps(response_json.get('error', response_json), indent=2)
            if run_id and record_id:
                full_error = f"{context + ' ' if context else ''}Error:\n{error_msg}"
                log_llm_error(run_id, record_id, full_error, error_log_dir)
            
            error_text = response_json.get('error', {}).get('message', 'Unknown error') if 'error' in response_json else 'No response from model'
            return {'content': f"Error: {error_text}", 'reasoning': None, 'reasoning_details': None}, {}
            
        except requests.Timeout:
            if attempt < MAX_RETRIES:
                print(f"⚠️  Timeout (attempt {attempt + 1}/{MAX_RETRIES + 1}) - Run: {run_id} - {context or 'Request'}")
                time.sleep(RETRY_BACKOFF_FACTOR * (2 ** attempt))
            else:
                if run_id and record_id:
                    error_msg = f"{context + ' ' if context else ''}Timeout after {MAX_RETRIES + 1} attempts"
                    log_llm_error(run_id, record_id, error_msg, error_log_dir)
                return {'content': f"Error: Request timeout after {MAX_RETRIES + 1} attempts", 'reasoning': None, 'reasoning_details': None}, {}
        except Exception as e:
            if run_id and record_id:
                error_msg = f"{context + ' ' if context else ''}Exception:\n{str(e)}"
                log_llm_error(run_id, record_id, error_msg, error_log_dir)
            return {'content': f"Error: {str(e)}", 'reasoning': None, 'reasoning_details': None}, {}

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
