import random
import string
import yaml

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def extract_config(config_module):
    """
    Extract all UPPERCASE config variables from a module and return as a dict.
    
    Args:
        config_module: The config module to extract from
    
    Returns:
        dict: Config variables with lowercase keys
    """
    config = {}
    for key in dir(config_module):
        if key.isupper() and not key.startswith('_'):
            value = getattr(config_module, key)
            config[key.lower()] = value
    return config


def load_prompts():
    with open('prompts.yaml', 'r') as f:
        prompts = yaml.safe_load(f)
    return prompts['debater_prompt_template'], prompts['private_reasoning_prompt'], prompts['action_prompt_template']
