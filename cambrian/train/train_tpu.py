import urllib3.util.retry

def infinite_is_exhausted(self):
    return False
# Monkey-patch the is_exhausted method
urllib3.util.retry.Retry.is_exhausted = infinite_is_exhausted


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
