# QA Evaluation Configuration

# Dataset configuration
DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
DATASET_SPLIT = "train"

# Model configuration
MODEL_NAME = "openai/gpt-4o-mini" 
# MODEL_NAME = "x-ai/grok-4-fast"
TEMPERATURE = 0.0

# Evaluation parameters
NUM_QUESTIONS = 198
RANDOM_SEED = 42 
NUM_CHOICES = 2  

# Threading configuration
MAX_THREADS = 5000

