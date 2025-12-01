# DATASET_NAME = "Idavidrein/gpqa"
# DATASET_SUBSET = "gpqa_diamond"
# DATASET_SPLIT = "train"

# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"

DATASET_NAME = "m-a-p/SuperGPQA"
DATASET_SUBSET = None
DATASET_SPLIT = "train"
# additional filters that can be applied
DATASET_FILTERS = {  
    'field': 'Physics',
    'difficulty': 'hard'
}

# MODEL_NAME = "openai/gpt-4o-mini" 
MODEL_NAME = "x-ai/grok-4-fast" 
# MODEL_NAME = "openai/gpt-3.5-turbo"
# MODEL_NAME = "qwen/qwen-2.5-7b-instruct" 
# MODEL_NAME = "meta-llama/llama-3-8b-instruct"
# MODEL_NAME = "qwen/qwen3-8b"
# MODEL_NAME = "qwen/qwen3-235b-a22b"

# MODEL_NAME = "deepseek/deepseek-v3.1-terminus"
# MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b"

# MODEL_NAME = "openai/o3"
# MODEL_NAME = "openai/gpt-oss-120b"

# MODEL_NAME = "google/gemini-3-pro-preview"

TEMPERATURE = 0.0

MAX_TOKENS = 15000
REASONING_EFFORT = None # 'low'
REASONING_MAX_TOKENS = None

NUM_QUESTIONS = 1
RANDOM_SEED = 42 
NUM_CHOICES = 2

RERUN = False  # will rerun a question/choice set even if it already exists
LENIENT_PARSING = False

SPECIFIC_QUESTION_IDXS = None # [183] # [167] # List of specific question indices to run QA for, e.g., [0, 5, 10]

MAX_THREADS = 2000
