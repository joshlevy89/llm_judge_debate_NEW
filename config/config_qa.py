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
MODEL_NAME = "meta-llama/llama-3-8b-instruct"

TEMPERATURE = 0.0

NUM_QUESTIONS = 198
RANDOM_SEED = 42 
NUM_CHOICES = 2  

MAX_THREADS = 20

SPECIFIC_QUESTION_IDXS = None # [167] # List of specific question indices to run QA for, e.g., [0, 5, 10]

