MODEL_NAME = 'x-ai/grok-4-fast'
TEMPERATURE = 0.0
MAX_THREADS = 100

LEAK_TYPES = {
    1: "Explicitly stating or implying that their answer is incorrect or weak",
    2: "Explicitly stating or impying that another answer/debater is correct",
    3: "Revising statements after unsatisfactory results/conclusions (e.g. 'but wait' or 'actually' after calculating a result that matches a different option, stating only after the fact that a calculation leading to an unsatisfactory result (e.g. the true correct answer) was misguided, saying something like 'I made a mistake', etc.)",
    4: "Other types of leaks not listed here"
}

