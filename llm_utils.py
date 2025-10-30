import re
import json
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config_general import REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF_FACTOR

def _get_session():
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

_session = _get_session()

def log_llm_error(run_id, record_id, error_message, error_log_dir='results/debate/error_logs'):
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

def _make_openrouter_request(prompt, model_name, api_key, temperature=0.0, reasoning_effort=None, reasoning_max_tokens=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    if reasoning_effort or reasoning_max_tokens:
        reasoning_config = {}
        if reasoning_effort:
            reasoning_config["effort"] = reasoning_effort
        if reasoning_max_tokens:
            reasoning_config["max_tokens"] = reasoning_max_tokens
        data["reasoning"] = reasoning_config
    
    response = _session.post(url, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
    return response.json()

def call_openrouter(prompt, model_name, api_key, temperature=0.0, reasoning_effort=None, reasoning_max_tokens=None, run_id=None, record_id=None, context=None, error_log_dir='results/debate/error_logs'):
    try:
        response_json = _make_openrouter_request(prompt, model_name, api_key, temperature, reasoning_effort, reasoning_max_tokens)
        
        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content'], response_json.get('usage', {})
        
        error_msg = json.dumps(response_json.get('error', response_json), indent=2)
        if run_id and record_id:
            full_error = f"{context + ' ' if context else ''}Error:\n{error_msg}"
            log_llm_error(run_id, record_id, full_error, error_log_dir)
        
        error_text = response_json.get('error', {}).get('message', 'Unknown error') if 'error' in response_json else 'No response from model'
        return f"Error: {error_text}", {}
        
    except Exception as e:
        if run_id and record_id:
            error_msg = f"{context + ' ' if context else ''}Exception:\n{str(e)}"
            log_llm_error(run_id, record_id, error_msg, error_log_dir)
        return f"Error: {str(e)}", {}

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

def parse_model_response(response_text):
    parsed = {
        'is_valid': False,
        'answer': None,
        'confidence': None,
        'reasoning': None
    }
    
    final_answer_match = re.search(r'<BEGIN FINAL ANSWER>(.*?)</END FINAL ANSWER>', response_text, re.DOTALL | re.IGNORECASE)
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
