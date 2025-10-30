import random

def parse_gpqa_item(item):
    question = item.get('Question')
    correct_answer = item.get('Correct Answer')
    all_choices = [
        item.get('Correct Answer'),
        item.get('Incorrect Answer 1'),
        item.get('Incorrect Answer 2'),
        item.get('Incorrect Answer 3')
    ]
    all_choices = [c.strip() for c in all_choices if c is not None]
    correct_answer = correct_answer.strip() if correct_answer else None
    return question, correct_answer, all_choices

def parse_mmlu_pro_item(item):
    question = item.get('question')
    options = item.get('options', [])
    answer_index = item.get('answer_index')
    all_choices = [opt.strip() for opt in options if opt]
    correct_answer = all_choices[answer_index] if answer_index is not None and answer_index < len(all_choices) else None
    return question, correct_answer, all_choices

def select_questions_and_options(dataset_name, dataset, num_questions, num_choices, seed):
    # Use seed to select questions
    rng_questions = random.Random(seed)
    total_questions = len(dataset)
    question_indices = rng_questions.sample(range(total_questions), min(num_questions, total_questions))
    
    results = []
    for idx in question_indices:
        item = dataset[idx]
        
        # Parse dataset item based on dataset name
        if dataset_name == "Idavidrein/gpqa":
            question, correct_answer, all_choices = parse_gpqa_item(item)
        elif dataset_name == "TIGER-Lab/MMLU-Pro":
            question, correct_answer, all_choices = parse_mmlu_pro_item(item)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Use question index as seed for reproducible option selection
        rng_options = random.Random(idx)
        
        # Select the number of choices requested
        if len(all_choices) >= num_choices:
            incorrect_choices = [c for c in all_choices if c != correct_answer]
            num_incorrect = num_choices - 1
            selected_incorrect = rng_options.sample(incorrect_choices, num_incorrect)
            selected_options = [correct_answer] + selected_incorrect
        else:
            selected_options = all_choices
        
        # Shuffle the selected options
        rng_options.shuffle(selected_options)
        
        # Find correct answer position
        correct_idx = selected_options.index(correct_answer)
        
        results.append({
            'question': question,
            'options': selected_options,
            'correct_idx': correct_idx,
            'original_idx': idx
        })
    
    return results

def format_options(options):
    options_text = ""
    for i, option in enumerate(options):
        options_text += f"{i}. {option}\n"
    return options_text.strip()