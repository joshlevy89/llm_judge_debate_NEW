from utils.debate_utils import *
import argparse

def __main__():
    import os
import re
import json
from pandas.core.strings.accessor import NoNewAttributesMixin
import yaml
import random
import string
import time
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from dotenv import load_dotenv
import config.config_debate as config_debate
from config.config_debate import (
    DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT,
    DEBATER_MODEL, DEBATER_TEMPERATURE,
    DEBATER_REASONING_EFFORT, DEBATER_REASONING_MAX_TOKENS,
    NUM_QUESTIONS, RANDOM_SEED, NUM_CHOICES, NUM_TURNS,
    PRIVATE_SCRATCHPAD, MAX_THREADS, MAX_OUTPUT_TOKENS,
    PUBLIC_ARGUMENT_WORD_LIMIT, PRIVATE_REASONING_WORD_LIMIT,
    LENIENT_PARSING_ARGUMENT
)
from utils.llm_utils import call_openrouter, get_openrouter_key_info, log_progress
from utils.dataset_utils import select_questions_and_options, format_options
from utils.debate_utils import *
from utils.shared_utils import extract_config, generate_run_id
import os


def setup_output_path(run_id):
    output_dir = Path('results') / 'human'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{run_id}.jsonl'

def main():
    # Add  
    parser = argparse.ArgumentParser()
    parser.add_argument('question_idx', type=int)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")
    
    run_id = generate_run_id()
    run_datetime = datetime.now().isoformat()
    results_path = setup_output_path(run_id)
    config = extract_config(config_debate)
    
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)[DATASET_SPLIT]
    questions_data = select_questions_and_options(DATASET_NAME, dataset, 1, NUM_CHOICES, None, [args.question_idx])
    
    debater_template, private_reasoning_prompt = load_prompts()
    
    for i, q_data in enumerate(questions_data):
        
    
        record_id = generate_run_id()
        debate_history = []

        # DISPLAY THE QUESTION
        print('='*80)
        print(q_data['question'])
        print(q_data['options'])

        # RUN DEBATE
        debater_assignments = q_data['options']
        cur_debater_idx = 0
        for turn in range(NUM_TURNS):
            turn_response = run_debate_turn(turn, debater_assignments, cur_debater_idx, q_data['question'], debate_history, debater_template, private_reasoning_prompt, api_key, run_id, record_id)
            debate_history.append(turn_response)
            cur_debater_idx += 1
            cur_debater_idx = cur_debater_idx % len(debater_assignments) # cycle back
            
            print(f"{'='*80}\nDebater {turn_response['debater_idx']} (Turn {turn_response['turn']})\n{'='*80}\n")
            if turn_response['success']:
                print(turn_response['parsed_response']['public_argument'])
            else:
                print('ERROR ' * 80)
                print(turn_response)
            
            # Get the next action from the human
            while True:
                print("="*80)
                print("Actions: 'next', 'end', or '<debater_idx>: <message>'")
                print("="*80)
                message = input("> ").strip()
                
                if message == 'end':
                    action = message
                    break
                elif message == 'next':
                    action = message
                    break
                elif ':' in message:
                    parts = message.split(':', 1)
                    try:
                        cur_debater_idx = int(parts[0].strip())
                        action = parts[1].strip()
                        break
                    except ValueError:
                        print("Invalid format. Use: <debater_idx>: <message>")
                else:
                    print("Invalid action.")

            debate_history.append({
                'persona': 'judge',
                'action': action,
                'is_human': True
            })
            
            if action == 'end':
                break

        # Potentially, at some point, add ability to input the verdict and do proper saving of the transcript
        print("="*80)
        print("Final Answer (index): ")
        verdict = input("> ").strip()
        print(f'Verdict: {verdict}')
        print(f"Correct Idx {q_data['correct_idx']}")


if __name__ == "__main__":
    main()

