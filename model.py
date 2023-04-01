import constants
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Hand:
    def __init__(self, cards: torch.Tensor):
        assert cards.shape[1:] == (13, 4, 13)
        self.cards = cards
        self.num_batch = cards.shape[0]


@dataclass
class DiscardPile:
    def __init__(self, discard_pile: torch.Tensor):
        assert discard_pile.shape[1:] == (constants.num_discard_pile_cards, 4, 13)
        self.num_batch = discard_pile.shape[0]
        self.discard_pile = discard_pile
        self.num_discard_pile_cards = np.zeros(self.num_batch, dtype=np.int32)

        for i in range(0, self.num_batch):
            for j in range(0, constants.num_discard_pile_cards):
                if self.discard_pile[i, j, :, :].sum() == 0:
                    self.num_discard_pile_cards[i] = j
                    break

        self.mask = self.num_discard_pile_cards > 0
        masked_discard_pile = self.discard_pile[self.mask, :, :, :]

        self.packed_discard_pile: Optional[torch.nn.utils.rnn.PackedSequence] = None
        if masked_discard_pile.shape[0] > 0:
            self.packed_discard_pile = torch.nn.utils.rnn.pack_padded_sequence(
                masked_discard_pile.view(-1, constants.num_discard_pile_cards, 4 * 13),
                lengths=torch.tensor(self.num_discard_pile_cards[self.mask]),
                batch_first=True,
                enforce_sorted=False,
            )


@dataclass
class GameState:
    def __init__(self, hand: Hand, discard_pile: DiscardPile):
        assert hand.num_batch == discard_pile.num_batch

        self.num_batch = hand.num_batch
        self.hand = hand
        self.discard_pile = discard_pile

    def gen_action_mask(self):
        mask = torch.zeros(self.num_batch, 13 * 4 * 13, 2, dtype=torch.bool)
        for i in range(0, self.num_batch):
            cards_mask = self.hand.cards[i, :, :, :].view(-1) > 0
            mask[i, cards_mask, 0] = True
            if self.discard_pile.num_discard_pile_cards[i] > 0:
                mask[
                    i,
                    cards_mask,
                    1,
                ] = True
        return mask.view(self.num_batch, -1)


class JinBeerDQN(nn.Module):
    def __init__(self):
        super(JinBeerDQN, self).__init__()
        self.hand_fc1 = nn.Linear(13 * 4 * 13, constants.num_actions * 2)
        self.hand_fc2 = nn.Linear(constants.num_actions * 2, constants.num_actions)
        self.discard_pile_gru = nn.GRU(4 * 13, constants.num_actions * 2, batch_first=True)
        self.discard_pile_fc1 = nn.Linear(constants.num_actions * 2, constants.num_actions)

    def forward(self, input: GameState):
        x_hand = input.hand.cards.view(input.num_batch, -1)
        assert x_hand.shape[1] == 13 * 4 * 13

        x_hand = F.relu(self.hand_fc1(x_hand))
        y = self.hand_fc2(x_hand)

        if input.discard_pile.packed_discard_pile is not None:
            x_discard_pile, _ = self.discard_pile_gru(input.discard_pile.packed_discard_pile)
            x_discard_pile = torch.stack(
                [x[-1, :] for x in torch.nn.utils.rnn.unpack_sequence(x_discard_pile)]
            )
            x_discard_pile = self.discard_pile_fc1(x_discard_pile)

            y[input.discard_pile.mask, :] = (
                0.3 * y[input.discard_pile.mask, :] + 0.7 * x_discard_pile
            )

        return y
