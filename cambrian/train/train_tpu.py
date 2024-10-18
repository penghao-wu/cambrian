import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Define your custom session creator
def create_session_with_retries():
    retry_strategy = Retry(
        total=500,  # Adjust the number of retries as needed
        backoff_factor=0.5,  # Increase delay between retries
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


from cambrian.train.train_fsdp import train

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import wandb
wandb.login(key='618eb3b78242f01000855a123d29e2ac98a60f30')
if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.distributed.xla_multiprocessing as xmp
    mp.set_start_method('spawn', force=True)
    xmp.spawn(train, args=(None,))
