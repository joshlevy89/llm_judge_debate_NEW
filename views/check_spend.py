import os
import requests
from dotenv import load_dotenv

def get_openrouter_key_info(api_key):
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching key info: {e}")
    return None

def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        return
    
    key_info = get_openrouter_key_info(api_key)
    
    if key_info:
        usage = key_info.get('data', {}).get('usage', 0)
        print(f"Total OpenRouter Spend: ${usage:.2f}")
    else:
        print("Could not retrieve spend information")

if __name__ == "__main__":
    main()

