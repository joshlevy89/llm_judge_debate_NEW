# JUDGE_MODEL = "openai/gpt-4o-mini"
# JUDGE_MODEL = "openai/gpt-3.5-turbo"
# JUDGE_MODEL = "qwen/qwen-2.5-7b-instruct"
# JUDGE_MODEL = "qwen/qwen-2.5-72b-instruct"
# JUDGE_MODEL = "qwen/qwen3-8b"
# JUDGE_MODEL = "qwen/qwen3-14b"
# JUDGE_MODEL = "qwen/qwen3-32b"
# JUDGE_MODEL = "qwen/qwen3-235b-a22b"
# JUDGE_MODEL = "meta-llama/llama-3-8b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.1-70b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.1-405b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
# JUDGE_MODEL = "meta-llama/llama-4-scout"
# JUDGE_MODEL = "meta-llama/llama-4-maverick"
# JUDGE_MODEL = "deepseek/deepseek-r1-distill-qwen-14b"
JUDGE_MODEL = "google/gemma-3-12b-it"
# JUDGE_MODEL = "google/gemma-3-27b-it"


DEBATE_RUN_ID = "17zguxe" # 2 choice gpqa
# DEBATE_RUN_ID = "q6wpwb7" # 4 choice gpqa

JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING_EFFORT = None
JUDGE_REASONING_MAX_TOKENS = None
MAX_OUTPUT_TOKENS = 5000

SKIP_QA = False
RERUN = False

SUBSET_N = None # Only run the verdict for the first N debates
SPECIFIC_RECORD_IDS = None#['wi4kbip'] # List of specific record_ids to run the verdict for

MAX_THREADS = 20  
