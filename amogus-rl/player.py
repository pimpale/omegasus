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
    def play(self, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class ActorPlayer(Player):
    def __init__(
            self,
            impostor_actor: network.Actor,
            impostor_critic: network.Critic,
            impostor_step: int,
            crewmate_actor: network.Actor,
            crewmate_critic: network.Critic,
            crewmate_step: int,
        ) -> None:
        self.impostor_actor = impostor_actor
        self.impostor_critic = impostor_critic
        self.impostor_step = impostor_step
        self.crewmate_actor = crewmate_actor
        self.crewmate_critic = crewmate_critic
        self.crewmate_step = crewmate_step

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action]:      
        obs = e.observe(player)
        actor = self.impostor_actor if obs.self_is_impostor else self.crewmate_actor

        device = network.deviceof(actor)

        action_probs = actor.forward(network.obs_to_tensor(obs, device))[
            0].to("cpu").detach().numpy()

        if np.isnan(action_probs).any():
            raise ValueError("NaN found!")

        action_entropy = scipy.stats.entropy(action_probs)

        if action_entropy < 0.001:
            raise ValueError("Entropy is too low!")

        legal_mask = e.legal_mask(player)

        raw_p = action_probs*legal_mask
        p = raw_p/np.sum(raw_p)

        chosen_action = env.Action(np.random.choice(len(p), p=p))

        return (
            obs,
            action_probs,
            chosen_action,
        )

    def name(self) -> str:
        return f"nn_ckpt_crewmate{self.crewmate_step}_impostor{self.impostor_step}"


class RandomPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action]:
        obs = e.observe(player)
        legal_mask = e.legal_mask(player)
        action_prob = scipy.special.softmax(
            np.random.random(size=len(legal_mask)))
        chosen_action: env.Action = np.argmax(action_prob*legal_mask)

        return (
            obs,
            action_prob,
            chosen_action,
        )

    def name(self) -> str:
        return "random"


class WaitPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, player: env.Player, e: env.Env) -> tuple[env.Observation, np.ndarray, env.Action]:
        obs = e.observe(player)

        chosen_action = env.Actions.WAIT
        action_prob = np.zeros(env.ACTION_SPACE_SIZE)

        return (
            obs,
            action_prob,
            chosen_action,
        )

    def name(self) -> str:
        return "random"
