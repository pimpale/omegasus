from abc import ABC, abstractmethod
import numpy as np
import scipy.special
import scipy.stats

import env
import network


class Player(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ActorPlayer(Player):
    def __init__(self, actor: network.Actor, critic: network.Critic, epoch: int) -> None:
        self.actor = actor
        self.critic = critic
        self.epoch = epoch

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(player)

        device = network.deviceof(self.actor)

        action_probs = self.actor.forward(network.obs_to_tensor(obs, device))[
            0].to("cpu").detach().numpy()

        action_entropy = scipy.stats.entropy(action_probs)
        if action_entropy < 0.001:
            raise ValueError("Entropy is too low!")

        if np.isnan(action_probs).any():
            raise ValueError("NaN found!")

        legal_mask = e.legal_mask(player)

        raw_p = action_probs*legal_mask
        p = raw_p/np.sum(raw_p)

        chosen_action = env.Action(np.random.choice(len(p), p=p))
        reward = e.play(chosen_action, player)

        return (
            obs,
            action_probs,
            chosen_action,
            reward
        )

    def name(self) -> str:
        return f"actor_ckpt_{self.epoch}"


class RandomPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(player)
        legal_mask = e.legal_mask(player)
        action_prob = scipy.special.softmax(
            np.random.random(size=len(legal_mask)))
        chosen_action: env.Action = np.argmax(action_prob*legal_mask)
        reward = e.play(chosen_action, player)

        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )

    def name(self) -> str:
        return "random"


class WaitPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action, env.Reward]:
        obs = e.observe(player)

        chosen_action = env.Actions.WAIT
        reward = e.play(chosen_action, player)
        action_prob = np.zeros(env.ACTION_SPACE_SIZE)

        return (
            obs,
            action_prob,
            chosen_action,
            reward
        )

    def name(self) -> str:
        return "random"
