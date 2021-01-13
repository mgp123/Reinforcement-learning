from random import seed

import torch

from src.garbage import ppo

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(2020)
    torch.manual_seed(2020)
    ppo()
