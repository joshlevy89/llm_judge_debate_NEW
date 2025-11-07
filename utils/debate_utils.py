def format_debate_history(history, show_private=False):
    if not history:
        return ""
    
    text = ""
    for entry in history:
        text += f"{'-'*80}\nDebater {entry['debater_idx']} (Turn {entry['turn']})\n{'-'*80}\n"
        
        if show_private and entry.get('internal_model_reasoning') is not None:
            text += f"[BEGIN INTERNAL REASONING]\n{entry['internal_model_reasoning']}\n[END INTERNAL REASONING]\n\n"
        
        if show_private and 'parsed_response' in entry and 'private_reasoning' in entry['parsed_response']:
            text += f"[BEGIN PRIVATE SCRATCHPAD REASONING]\n{entry['parsed_response']['private_reasoning']}\n[END PRIVATE SCRATCHPAD REASONING]\n\n"
        
        if 'parsed_response' in entry and 'public_argument' in entry['parsed_response']:
            text += f"[BEGIN PUBLIC ARGUMENT]\n{entry['parsed_response']['public_argument']}\n[END PUBLIC ARGUMENT]\n"
    
    return text

