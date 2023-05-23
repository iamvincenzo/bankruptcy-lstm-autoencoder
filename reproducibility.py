##############################################################################################################################################
# SOME REFERENCES: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy #
#############################################################################################################################################

###########################################################################
# SOME REFERENCES: https://pytorch.org/docs/stable/notes/randomness.html #
##########################################################################

import os
import torch
import random
import numpy as np

# """ Function used for results reproducibility. """
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

""" Callable objects. """
class SeedWorker:
    def __init__(self, worker_seed):
        self.worker_seed = worker_seed

    def __call__(self):
        np.random.seed(self.worker_seed)
        random.seed(self.worker_seed)

""" Function used for results reproducibility. """
def set_seed(seed=42):
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"\nRandom seed set as: {seed}")
