from ast import List
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

import numpy as np
import torch
import constants
from model import DiscardPile, Hand, JinBeerDQN, GameState


@dataclass
class MemoryShard:
    state: GameState
    action: int
    reward: float
    next_state: GameState


class JinAgent:
    def __init__(self, learning_rate=0.001):
        self.online_dqn = JinBeerDQN().to(dtype=constants.fp_type_torch, device=constants.device)
        self.target_dqn = JinBeerDQN().to(dtype=constants.fp_type_torch, device=constants.device)
        self.memory = deque(maxlen=20_000)

        self.optimizer = torch.optim.SGD(self.online_dqn.parameters(), lr=learning_rate)
        self.online_dqn.load_state_dict(self.target_dqn.state_dict())

    def act(self, state: GameState) -> int:
        qs = self.online_dqn(state)
        mask = state.gen_action_mask() == False
        qs[mask] = -np.inf
        return qs.argmax(dim=1).item()

    def remember(self, memory_shard: MemoryShard):
        self.memory.append(memory_shard)

    def __stack_states(batch: Iterable[MemoryShard]):
        hands_cards = torch.stack([x.state.hand.cards.squeeze(0) for x in batch])
        discard_piles_cards = torch.stack(
            [x.state.discard_pile.discard_pile.squeeze(0) for x in batch]
        )
        return GameState(Hand(hands_cards), DiscardPile(discard_piles_cards))

    def __stack_next_states(batch: Iterable[MemoryShard]):
        hands_cards = torch.stack([x.next_state.hand.cards.squeeze(0) for x in batch])
        discard_piles_cards = torch.stack(
            [x.next_state.discard_pile.discard_pile.squeeze(0) for x in batch]
        )
        return GameState(Hand(hands_cards), DiscardPile(discard_piles_cards))

    def learn(self, batch_size=32) -> float:
        if len(self.memory) < 5000:
            return 0.0

        batch = np.random.choice(self.memory, batch_size)
        states = JinAgent.__stack_states(batch)
        actions = (
            torch.tensor([x.action for x in batch])
            .unsqueeze(-1)
            .to(dtype=torch.int64, device=constants.device)
        )
        rewards = (
            torch.tensor([x.reward for x in batch])
            .unsqueeze(-1)
            .to(dtype=constants.fp_type_torch, device=constants.device)
        )
        next_states = JinAgent.__stack_next_states(batch)

        q_values = self.online_dqn(states)
        next_q_values = self.target_dqn(next_states)

        q_value = q_values.gather(1, actions).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + 0.99 * next_q_value

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_ems(self, tau):
        for key in self.target_dqn.state_dict().keys():
            self.target_dqn.state_dict()[key] = (
                tau * self.online_dqn.state_dict()[key]
                + (1.0 - tau) * self.target_dqn.state_dict()[key]
            )
