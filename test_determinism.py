import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

prompt = """Which of the following issues are the most common sources of difficult-to-spot erroneous results generated in genomics data analysis:

- Mutually incompatible data formats
- The "chr" / "no chr" confusion
- Reference assembly mismatch
- Incorrect ID conversion

Options: ['2, 3 and 4', '3 and 4', 'All of the above', '2 and 3']"""

N = 2

responses = []
for i in range(N):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "x-ai/grok-4-fast",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "seed": 42
        }
    )
    content = response.json()['choices'][0]['message']['content']
    responses.append(content)
    print(f"\n{'='*80}\nResponse {i+1}:\n{'='*80}")
    print(content)

print(f"\n{'='*80}\nDeterminism Check:\n{'='*80}")
if len(set(responses)) == 1:
    print("✓ All responses are identical (deterministic)")
else:
    print("✗ Responses differ (non-deterministic)")
    for i in range(len(responses)-1):
        if responses[i] != responses[i+1]:
            print(f"\nComparing Response {i+1} vs Response {i+2}:")
            s1, s2 = responses[i], responses[i+1]
            max_len = max(len(s1), len(s2))
            
            for j in range(max_len):
                c1 = s1[j] if j < len(s1) else '∅'
                c2 = s2[j] if j < len(s2) else '∅'
                if c1 != c2:
                    context_start = max(0, j - 20)
                    context_end = min(max_len, j + 20)
                    print(f"\nDifference at position {j}:")
                    print(f"  Response {i+1}: ...{s1[context_start:context_end]}...")
                    print(f"  Response {i+2}: ...{s2[context_start:context_end]}...")
                    print(f"  Char {i+1}: '{c1}' (ord={ord(c1) if c1 != '∅' else 'N/A'})")
                    print(f"  Char {i+2}: '{c2}' (ord={ord(c2) if c2 != '∅' else 'N/A'})")
                    break

