DEBATE_RUN_ID = "17zguxe"

# JUDGE_MODEL = "openai/gpt-4o-mini"
# JUDGE_MODEL = "openai/gpt-3.5-turbo"
JUDGE_MODEL = "qwen/qwen-2.5-7b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3-8b-instruct"
JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING_EFFORT = None
JUDGE_REASONING_MAX_TOKENS = None
MAX_OUTPUT_TOKENS = 5000

SUBSET_N = None # Only run the verdict for the first N debates
SPECIFIC_RECORD_IDS = None#['wi4kbip'] # List of specific record_ids to run the verdict for

MAX_THREADS = 20
