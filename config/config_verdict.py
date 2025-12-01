# JUDGE_MODEL = "openai/gpt-4o-mini"
# JUDGE_MODEL = "x-ai/grok-4-fast"
# JUDGE_MODEL = "openai/gpt-3.5-turbo"
# JUDGE_MODEL = "qwen/qwen-2.5-7b-instruct"
# JUDGE_MODEL = "qwen/qwen-2.5-72b-instruct"
# JUDGE_MODEL = "qwen/qwen3-8b"
# JUDGE_MODEL = "qwen/qwen3-14b"
# JUDGE_MODEL = "qwen/qwen3-32b"
# JUDGE_MODEL = "qwen/qwen3-235b-a22b"
# JUDGE_MODEL = "meta-llama/llama-3-8b-instruct"
JUDGE_MODEL = "meta-llama/llama-3.1-8b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.1-70b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.1-405b-instruct"
# JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
# JUDGE_MODEL = "meta-llama/llama-4-scout"
# JUDGE_MODEL = "meta-llama/llama-4-maverick"
# JUDGE_MODEL = "deepseek/deepseek-r1-distill-qwen-14b"
# JUDGE_MODEL = "google/gemma-3-12b-it"
# JUDGE_MODEL = "google/gemma-3-27b-it"
# JUDGE_MODEL = "anthropic/claude-3.5-haiku"

# JUDGE_MODEL = "deepseek/deepseek-r1-distill-llama-70b"
# JUDGE_MODEL = "deepseek/deepseek-r1-distill-qwen-14b"
# JUDGE_MODEL = "deepseek/deepseek-chat"
# JUDGE_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
# JUDGE_MODEL = "deepseek/deepseek-r1-distill-qwen-32b"
# JUDGE_MODEL = "deepseek/deepseek-chat-v3-0324"
# JUDGE_MODEL = "deepseek/deepseek-r1"
# JUDGE_MODEL = "deepseek/deepseek-chat-v3.1"
# JUDGE_MODEL = "deepseek/deepseek-v3.2-exp"
# JUDGE_MODEL = "deepseek/deepseek-v3.1-terminus"


# DEBATE_RUN_ID = "17zguxe" # 2 choice gpqa
# DEBATE_RUN_ID = "q6wpwb7" # 4 choice gpqa
# DEBATE_RUN_ID = "z1qae3w" # 4 choice gpqa - deepseek/deepseek-v3.1-terminus as debater
# DEBATE_RUN_ID = "7q0xvsw"  # 4 choice gpqa - o3 (low)
# DEBATE_RUN_ID = "7i24jh4"  # 4 choice gpqa - gpt oss 120b (high)
# DEBATE_RUN_ID = "1v72vdi"  # 4 choice gpqa - gpt-4o-mini as judge (control)
# DEBATE_RUN_ID = "zye2tmr" # sequential run with 10 turns and two debaters
# DEBATE_RUN_ID = "rdf775j" # sequential run with 10 turns, two debaters, an interactive judge that can do next, end, or ask open ended questions (tends to use this option)
# DEBATE_RUN_ID = "nft6lsw"  # sequential run with 10 turns, two debaters, non-interative, UPDATED PROMPTS, GEMINI-3-Pro-Preview AS DEBATER
# DEBATE_RUN_ID = "kxsb23b"  # sequential run with 10 turns, two debaters, non-interative, UPDATED PROMPTS, grok-4-fast as debater (validation set of the 30 ids)
# DEBATE_RUN_ID = "n27oezk"   # sequential run with 10 turns, two debaters, non-interative, UPDATED PROMPTS, gpt-4o-mini as debater (validation set of the 30 ids). control.
# DEBATE_RUN_ID = "o5g5vzf" # sequential run with 10 turns, two debaters, non-interative, UPDATED PROMPTS, grok-4-fast as debater (test set of the 30 ids)
# DEBATE_RUN_ID = "d97mc4l"   # sequential run with 10 turns, two debaters, non-interative, UPDATED PROMPTS, gpt-4o-mini as debater (test set of the 30 ids). control.
DEBATE_RUN_ID = "tw3w3f9"  # all 198, grok-4-fast 10 turns updated
# DEBATE_RUN_ID = "5mflq0e"  # all 198, gpt-oss-120b (high) 10 turns updated prompts
# DEBATE_RUN_ID = "q9uj26k" # 4 way debate, all 198, gpt-oss-120b (high) 10 turns, updated prompts
# DEBATE_RUN_ID = 'i3mvjzg' # # 4 way debate, all 198, grok-4-fast 10 turns, updated prompts
# DEBATE_RUN_ID = "human"

JUDGE_TEMPERATURE = 0.0
JUDGE_REASONING_EFFORT = None
JUDGE_REASONING_MAX_TOKENS = None
MAX_OUTPUT_TOKENS = 5000

SKIP_QA = False
RERUN = False

SUBSET_N = None # Only run the verdict for the first N debates
SPECIFIC_RECORD_IDS = None # ['ezrn67r'] # List of specific record_ids to run the verdict for
UPTO_TURNS = None


MAX_THREADS = 200  
