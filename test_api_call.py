import os
import sys
import logging

# Add the parent directory to the sys.path to allow importing vector_search_service
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Manual .env loader (copied from build_db.py)
def manual_load_dotenv(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key, value = parts
                    if value.startswith(('"', "'")) and value.endswith(("'", "'")):
                        if value[0] == value[-1]:
                            value = value[1:-1]
                    os.environ[key] = value
    except FileNotFoundError:
        logging.error(f"Dotenv file not found at {path}")
    except Exception as e:
        logging.error(f"An error occurred during manual .env loading: {e}")

# Load .env variables
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
manual_load_dotenv(dotenv_path)

# Configure logging to show all messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

import vector_search_service

if __name__ == "__main__":
    law_mst_to_test = "268551" # 산업안전보건법
    logging.info(f"Testing fetch_law_details for law_mst: {law_mst_to_test}")
    try:
        result = vector_search_service.fetch_law_details(law_mst_to_test)
        logging.info(f"Fetch result for {law_mst_to_test}: {result}")
        if not result:
            logging.warning(f"No articles fetched for {law_mst_to_test}. Check API response or parsing logic.")
    except Exception as e:
        logging.error(f"Error calling fetch_law_details for {law_mst_to_test}: {e}")
