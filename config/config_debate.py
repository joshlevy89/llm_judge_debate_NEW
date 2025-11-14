DEBATER_MODEL = "x-ai/grok-4-fast"
# DEBATER_MODEL = "google/gemini-2.5-flash-lite" # too many tokens and as a result often doesn't output Begin Public Argument tag. Also screws up tags in general but that part can be fixed with lenient parsing.
# DEBATER_MODEL = "deepseek/deepseek-v3.1-terminus"  
# DEBATER_MODEL = "deepseek/deepseek-v3.2-exp"
# DEBATER_MODEL = "google/gemini-2.5-flash" # it leaks too much - it does help to turn up debater temperature but it still leaks too much
# DEBATER_MODEL = "google/gemini-2.5-pro" # too expensive - by the way it's thinking by default.
# DEBATER_MODEL = "anthropic/claude-sonnet-4.5" # it's decent. not too expensive, can get the tags right with my updated prompt. using private scratchpad. but it leaked 2/5. could be viable though.
# DEBATER_MODEL = "openai/o3"
# DEBATER_MODEL = "openai/gpt-oss-120b" 


DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
DATASET_SPLIT = "train"

# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"

DEBATER_TEMPERATURE = 0.3

# MAX_OUTPUT_TOKENS = 125000
# DEBATER_REASONING_EFFORT = 'high'
# DEBATER_REASONING_MAX_TOKENS = None

MAX_OUTPUT_TOKENS = 5000
DEBATER_REASONING_EFFORT = None
DEBATER_REASONING_MAX_TOKENS = None

PRIVATE_SCRATCHPAD = False

# PUBLIC_ARGUMENT_WORD_LIMIT = 100
# PRIVATE_REASONING_WORD_LIMIT = 150

PUBLIC_ARGUMENT_WORD_LIMIT = 200
PRIVATE_REASONING_WORD_LIMIT = 1000

LENIENT_PARSING_ARGUMENT = False

NUM_QUESTIONS = 1
RANDOM_SEED = 42
NUM_CHOICES = 2
NUM_TURNS = 100

MAX_THREADS = 5000

DEBATE_MODE = "sequential"