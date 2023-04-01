import numpy as np
import torch


device = "cuda"
fp_type_np = np.float32
fp_type_torch = torch.float32
num_discard_pile_cards = 96
num_actions = 13 * 4 * 13 * 2
