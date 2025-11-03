DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
DATASET_SPLIT = "train"

# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"

# MODEL_NAME = "openai/gpt-4o-mini" 
# MODEL_NAME = "x-ai/grok-4-fast" 
# MODEL_NAME = "openai/gpt-3.5-turbo"
# MODEL_NAME = "qwen/qwen-2.5-7b-instruct" 
# MODEL_NAME = "meta-llama/llama-3-8b-instruct"
# MODEL_NAME = "qwen/qwen3-8b"
MODEL_NAME = "qwen/qwen3-235b-a22b"

TEMPERATURE = 0.0

NUM_QUESTIONS = 198
RANDOM_SEED = 42 
NUM_CHOICES = 4

RERUN = False  # will rerun a question/choice set even if it already exists

SPECIFIC_QUESTION_IDXS = None # [167] # List of specific question indices to run QA for, e.g., [0, 5, 10]

MAX_THREADS = 20
