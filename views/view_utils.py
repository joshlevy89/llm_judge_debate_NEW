def shorten_name(name):
    if not name or not isinstance(name, str):
        return name
    parts = name.split('/')
    return parts[-1] if len(parts) > 1 else name

