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

