import numpy as np
import torch
import visdom
from agent import JinAgent, MemoryShard
from env import JinEnv


class JinTrainer:
    def __init__(
        self,
        env: JinEnv,
        agent1: JinAgent,
        agent2: JinAgent,
        sigma_start=0.9,
        sigma_decay=1000,
        sigma_end=0.05,
    ) -> None:
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.sigma_start = sigma_start
        self.sigma_decay = sigma_decay
        self.sigma_end = sigma_end
        self.tau = 0.005
        self.visdom = visdom.Visdom()

    def agent(self, hand):
        if hand == 1:
            return self.agent1
        elif hand == 2:
            return self.agent2
        else:
            raise ValueError("Invalid hand")

    @staticmethod
    def cycle_hand(hand):
        if hand == 1:
            return 2
        elif hand == 2:
            return 1
        else:
            raise ValueError("Invalid hand")

    def train(self, n_episodes=1000):
        episodes = []
        rewards_agent1 = []
        rewards_agent2 = []

        best_reward = -1000
        overall_steps_done = 0
        for episode in range(0, n_episodes):
            episode_agent1_reward = 0
            episode_agent2_reward = 0
            episode_agent1_loss = 0
            episode_agent2_loss = 0

            hand = 1
            current_state = self.env.state(hand)
            self.env.reset()
            episode_steps_done = 0
            done = False
            while not done:
                current_agent = self.agent(hand)

                sigma = self.sigma_end + (self.sigma_start - self.sigma_end) * np.exp(
                    -1.0 * overall_steps_done / self.sigma_decay
                )
                if np.random.random() < sigma:
                    action: int = np.random.choice(self.env.action_space(hand))
                else:
                    action: int = current_agent.act(current_state)

                next_state, reward, done = self.env.step(action, hand)
                current_agent.remember(MemoryShard(current_state, action, reward, next_state))
                loss = current_agent.learn()

                if hand == 1:
                    episode_agent1_reward += reward
                    episode_agent1_loss += loss
                elif hand == 2:
                    episode_agent2_reward += reward
                    episode_agent2_loss += loss

                self.agent1.update_ems(self.tau)
                self.agent2.update_ems(self.tau)
                current_state = next_state
                hand = JinTrainer.cycle_hand(hand)
                overall_steps_done += 1
                episode_steps_done += 1

            if episode_agent1_reward > best_reward:
                torch.save(self.agent1.online_dqn.state_dict(), f"best.pth")
                best_reward = episode_agent1_reward

            if episode_agent2_reward > best_reward:
                torch.save(self.agent2.online_dqn.state_dict(), f"best.pth")
                best_reward = episode_agent2_reward

            episodes.append(episode)
            rewards_agent1.append(episode_agent1_reward)
            rewards_agent2.append(episode_agent2_reward)

            self.visdom.line(
                X=np.array(episodes),
                Y=np.array(rewards_agent1),
                win="agent1_rewards",
                update="append",
                opts=dict(title="Agent-1 Rewards", webgl=True),
            )

            self.visdom.line(
                X=np.array(episodes),
                Y=np.array(rewards_agent2),
                win="agent2_rewards",
                update="append",
                opts=dict(title="Agent-2 Rewards", webgl=True),
            )

            print(
                f"Episode {episode}:\n",
                f"Reward-1 = {episode_agent1_reward}. Reward-2 = {episode_agent2_reward}\n"
                f"Loss-1 = {episode_agent1_loss / episode_steps_done}. Loss-2 = {episode_agent2_loss / episode_steps_done}\n",
            )
