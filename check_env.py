import os
import re

def manual_load_dotenv(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                match = re.match(r'^([^=]+)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    # Simpler quote stripping
                    if value.startswith(("", "'")) and value.endswith(("", "'")):
                        if value[0] == value[-1]:
                            value = value[1:-1]
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"Error: Dotenv file not found at {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Manually load the .env file
dotenv_path = "C:\\Users\\kwater\\Desktop\\safety-ai\\.env"
manual_load_dotenv(dotenv_path)

# Check for the API key
api_key = os.getenv("GEMINI_API_KEY")

# Print the result
if api_key:
    print(f"Success: Found the API key. (Starts with: {api_key[:4]}...)")
else:
    print("Failure: Could not find 'GEMINI_API_KEY' even after manual load.")