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
    def play(self, impostor: bool, obs: env.Observation) -> env.Action:
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

    def play(self, impostor: bool, obs: env.Observation) -> env.Action:
        actor = self.impostor_actor if impostor else self.crewmate_actor

        device = network.deviceof(actor)

        action_probs = (
            actor.forward(network.obs_to_tensor(obs, device)[0])[0]
            .to("cpu")
            .detach()
            .numpy()
        )

        chosen_action = env.Action(
            np.random.choice(env.ACTION_SPACE_SIZE, p=action_probs)
        )

        return chosen_action

    def name(self) -> str:
        return f"nn_ckpt_crewmate{self.crewmate_step}_impostor{self.impostor_step}"


class RandomPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, impostor: bool, obs: env.Observation) -> env.Action:
        return env.Action(np.random.choice(env.ACTION_SPACE_SIZE))

    def name(self) -> str:
        return "random"


class GreedyPlayer(Player):
    def __init__(self) -> None:
        pass

    def play(self, impostor: bool, obs: env.Observation) -> env.Action:
        my_location = env.OBS_XSIZE // 2, env.OBS_YSIZE // 2

        chosen_action = env.Actions.WAIT

        if impostor:
            # move towards nearest player if impostor
            crewmate_locations = np.argwhere(obs.view[env.CREWMATE_CHANNEL] == 1)
            if len(crewmate_locations) != 0:
                nearest_location = crewmate_locations[
                    np.argmin(np.linalg.norm(crewmate_locations - my_location, axis=1))
                ]
                if nearest_location[0] > my_location[0]:
                    chosen_action = env.Actions.MOVE_DOWN
                elif nearest_location[0] < my_location[0]:
                    chosen_action = env.Actions.MOVE_UP
                elif nearest_location[1] > my_location[1]:
                    chosen_action = env.Actions.MOVE_RIGHT
                elif nearest_location[1] < my_location[1]:
                    chosen_action = env.Actions.MOVE_LEFT
        else:
            # if crewmate move towards nearest task
            task_locations = np.argwhere(obs.view[env.TASK_CHANNEL] == 1)
            if len(task_locations) != 0:
                nearest_location = task_locations[
                    np.argmin(np.linalg.norm(task_locations - my_location, axis=1))
                ]
                if nearest_location[0] > my_location[0]:
                    chosen_action = env.Actions.MOVE_DOWN
                elif nearest_location[0] < my_location[0]:
                    chosen_action = env.Actions.MOVE_UP
                elif nearest_location[1] > my_location[1]:
                    chosen_action = env.Actions.MOVE_RIGHT
                elif nearest_location[1] < my_location[1]:
                    chosen_action = env.Actions.MOVE_LEFT

        return chosen_action

    def name(self) -> str:
        return "greedy"
