from cambrian.train.train_fsdp import train
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Define your custom session creator
def create_session_with_retries():
    retry_strategy = Retry(
        total=100,  # Adjust the number of retries as needed
        backoff_factor=0,  # Increase delay between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Save the original requests.get method
original_get = requests.get

# Define a new get method that uses the session with retries
def new_get(*args, **kwargs):
    session = create_session_with_retries()
    return session.get(*args, **kwargs)

# Monkey-patch requests.get
requests.get = new_get

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
