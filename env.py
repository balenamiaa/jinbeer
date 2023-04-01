from dataclasses import dataclass
import numpy as np
import torch

from model import DiscardPile, GameState, Hand

import constants


@dataclass
class Card:
    def __init__(self, suit: int, rank: int):
        self.suit = suit
        self.rank = rank

    def __iter__(self):
        yield self.suit
        yield self.rank


@dataclass
class StepResult:
    def __init__(self, state: GameState, reward: float, done: bool):
        self.state = state
        self.reward = reward
        self.done = done

    def __iter__(self):
        yield self.state
        yield self.reward
        yield self.done


class JinEnv:
    def __init__(self):
        self.hand_1 = JinEnv.__gen_random_hand()
        self.hand_2 = JinEnv.__gen_random_hand()
        self.discard_pile = np.zeros(
            (constants.num_discard_pile_cards, 4, 13), dtype=constants.fp_type_np
        )
        self.n_discard_pile = 0

    @staticmethod
    def __gen_random_hand() -> np.ndarray:
        hand = np.zeros((13, 4, 13), dtype=constants.fp_type_np)
        for i in range(13):
            rank = np.random.randint(0, 13)
            suit = np.random.randint(0, 4)
            hand[i, suit, rank] = 1
        return hand

    def value(self, hand) -> float:
        if hand == 1:
            cards_mat = self.hand_1
        elif hand == 2:
            cards_mat = self.hand_2
        else:
            raise ValueError("Invalid hand")
        possible_ranks = np.arange(0, 13)
        possible_suits = np.arange(0, 4)

        ranks = []
        suits = []
        for index in np.ndindex(cards_mat.shape):
            (i, j, k) = index
            if cards_mat[i, j, k] > 0:
                suits.append(j)
                ranks.append(k)

        rank_counts = {rank: np.count_nonzero(np.array(ranks) == rank) for rank in possible_ranks}
        suit_counts = {suit: np.count_nonzero(np.array(suits) == suit) for suit in possible_suits}

        value = 0

        for rank in possible_ranks:
            if rank_counts[rank] >= 3:
                rank_suits = []
                for index in np.ndindex(cards_mat.shape):
                    (i, j, k) = index
                    if cards_mat[i, j, k] > 0 and k == rank:
                        rank_suits.append(j)

                if len(rank_suits) == len(set(rank_suits)):
                    value += (rank + 1) * rank_counts[rank]

        for suit in possible_suits:
            if suit_counts[suit] >= 3:
                sequential_ranks = []
                for rank in possible_ranks:
                    if rank_counts[rank] > 0 and suits[ranks.index(rank)] == suit:
                        sequential_ranks.append(rank)
                sequential_ranks.sort()
                for i in range(len(sequential_ranks) - 2):
                    if (
                        sequential_ranks[i + 1] == sequential_ranks[i] + 1
                        and sequential_ranks[i + 2] == sequential_ranks[i] + 2
                    ):
                        value += sum(sequential_ranks[i : i + 3])
        return value

    def action_space(self, hand):
        if hand == 1:
            cards_mat = self.hand_1
        elif hand == 2:
            cards_mat = self.hand_2
        else:
            raise ValueError("Invalid hand")

        actions = []
        for i in range(13):
            possible_cards = cards_mat[i, :, :]
            maybe_card = np.where(possible_cards > 0)
            if len(maybe_card[0]) == 0:
                continue
            actions.append(
                np.ravel_multi_index((i, maybe_card[0][0], maybe_card[1][0], 0), (13, 4, 13, 2))
            )
            if self.n_discard_pile > 0:
                actions.append(
                    np.ravel_multi_index((i, maybe_card[0][0], maybe_card[1][0], 1), (13, 4, 13, 2))
                )
        return actions

    def reset(self):
        self.__init__()

    def remove_card(self, position, card: Card, hand):
        if hand == 1:
            cards_mat = self.hand_1
        elif hand == 2:
            cards_mat = self.hand_2
        else:
            raise ValueError("Invalid hand")
        cards_mat[position, *card] = 0

    def put_card(self, position, card: Card, hand):
        if hand == 1:
            cards_mat = self.hand_1
        elif hand == 2:
            cards_mat = self.hand_2
        else:
            raise ValueError("Invalid hand")
        cards_mat[position, *card] = 1

    def pull_random_card(self, position, hand):
        if hand == 1:
            cards_mat = self.hand_1
        elif hand == 2:
            cards_mat = self.hand_2
        else:
            raise ValueError("Invalid hand")
        suit = np.random.randint(0, 4)
        rank = np.random.randint(0, 13)
        cards_mat[position, suit, rank] = 1

    def push_discard_pile(self, card: Card):
        self.discard_pile[self.n_discard_pile, *card] = 1
        self.n_discard_pile += 1

    def pop_discard_pile(self):
        self.n_discard_pile -= 1
        card = np.where(self.discard_pile[self.n_discard_pile, :, :] > 0)
        self.discard_pile[self.n_discard_pile, :, :] = 0
        return Card(card[0][0], card[1][0])

    def take_action(self, action, hand):
        position, suit, rank, discard = np.unravel_index(action, (13, 4, 13, 2))
        card = Card(suit, rank)
        self.remove_card(position, card, hand)
        if discard == 0:
            self.pull_random_card(position, hand)
        else:
            new_card = self.pop_discard_pile()
            self.put_card(position, new_card, hand)
        self.push_discard_pile(card)

    def state(self, hand):
        if hand == 1:
            hand = self.hand_1
        elif hand == 2:
            hand = self.hand_2
        else:
            raise ValueError("Invalid hand")
        return GameState(
            Hand(
                torch.tensor(
                    hand, dtype=constants.fp_type_torch, device=constants.device
                ).unsqueeze(0)
            ),
            DiscardPile(
                torch.tensor(
                    self.discard_pile, dtype=constants.fp_type_torch, device=constants.device
                ).unsqueeze(0)
            ),
        )

    def step(self, action, hand):
        value = self.value(hand)
        self.take_action(action, hand)
        new_value = self.value(hand)

        if new_value > value:
            reward = 1
        elif new_value < value:
            reward = -1
        else:
            reward = 0

        done = self.n_discard_pile >= constants.num_discard_pile_cards
        return StepResult(self.state(hand), reward, done)
