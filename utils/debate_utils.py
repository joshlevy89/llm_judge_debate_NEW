import re

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


def parse_debater_response(response_text, private_scratchpad, lenient_argument_parsing=False):
    public_match = re.search(r'<BEGIN PUBLIC ARGUMENT>(.*?)</?END PUBLIC ARGUMENT>', response_text, re.DOTALL | re.IGNORECASE)
    
    if not public_match:
        return None, "Missing <BEGIN PUBLIC ARGUMENT> or </END PUBLIC ARGUMENT> tags"
    
    parsed = {'public_argument': public_match.group(1).strip()}
    
    if private_scratchpad:
        if lenient_argument_parsing:
            parts = response_text.split(public_match.group(0), 1)
            private_reasoning = parts[0].strip() if parts[0] else ""
            parsed['private_reasoning'] = private_reasoning
        else:
            private_match = re.search(r'<BEGIN PRIVATE REASONING>(.*?)</?END PRIVATE REASONING>', response_text, re.DOTALL | re.IGNORECASE)
            if not private_match:
                return None, "Missing <BEGIN PRIVATE REASONING> or </END PRIVATE REASONING> tags"
            parsed['private_reasoning'] = private_match.group(1).strip()
    
    return parsed, None