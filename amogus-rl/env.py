import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID
from gym import spaces
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias, Literal
import functools

# size of game board
BOARD_XSIZE = 5
BOARD_YSIZE = 5

MAX_STEPS = 25

# size of the action space
ACTION_SPACE_SIZE = 5  # U, D, L, R, W

# size of the observation (player is located at the center)
OBS_XSIZE = 5
OBS_YSIZE = 5
# impostors, crewmates, dead, tasks, wall
OBS_NUM_CHANNELS = 5

# Channel IDs
IMPOSTOR_CHANNEL = 0
CREWMATE_CHANNEL = 1
DEAD_CHANNEL = 2
TASK_CHANNEL = 3
WALL_CHANNEL = 4


# integer [0, 1, 2, 3, 4] <=> [U, D, L, R, W]
Action: TypeAlias = np.int8


# Action IDs
class Actions:
    MOVE_LEFT = Action(0)
    MOVE_RIGHT = Action(1)
    MOVE_UP = Action(2)
    MOVE_DOWN = Action(3)
    WAIT = Action(4)


@dataclass
class PlayerState:
    """State of a single player"""

    location: tuple[int, int]
    impostor: bool


Observation: TypeAlias = np.ndarray[int, np.dtype[np.bool_]]


@dataclass
class State:
    """Overall state of the game"""

    players: dict[AgentID, PlayerState]
    dead: np.ndarray[Any, np.dtype[np.int8]]
    tasks: np.ndarray[Any, np.dtype[np.int8]]


def print_action_description(action: Action) -> None:
    """Prints a description of the given action."""
    action_descriptions = {
        Actions.MOVE_LEFT: "Move Left",
        Actions.MOVE_RIGHT: "Move Right",
        Actions.MOVE_UP: "Move Up",
        Actions.MOVE_DOWN: "Move Down",
        Actions.WAIT: "Wait",
    }
    print(action_descriptions[action])


def print_obs(obs: np.ndarray):
    for y in range(OBS_YSIZE):
        q = ""
        for x in range(OBS_XSIZE):
            c = "⬛"
            if obs[DEAD_CHANNEL, y, x]:
                c = "💀"
            elif obs[IMPOSTOR_CHANNEL, y, x]:
                c = "👽"
            elif obs[CREWMATE_CHANNEL, y, x]:
                c = "🧑‍🚀"
            elif obs[WALL_CHANNEL, y, x]:
                c = "🧱"
            elif obs[TASK_CHANNEL, y, x]:
                c = "📦"
            q += c
        print(q)


class AmogusEnv(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "amogus_env"}

    def __init__(self, initial_state: State, render_mode=None):
        self.initial_state: State = initial_state
        self.state: State = initial_state
        self.agents = list(self.state.players.keys())
        self.possible_agents = list(self.state.players.keys())
        self.steps = 0

    def observe(self, agent: AgentID) -> Observation:
        """Observation by a single player of the game"""
        ax, ay = self.state.players[agent].location

        # how much to pad the observation with zeros
        padding = ((OBS_YSIZE // 2, OBS_YSIZE // 2), (OBS_XSIZE // 2, OBS_XSIZE // 2))

        # Calculate the padded versions of dead, tasks, and wall channels
        padded_dead = np.pad(self.state.dead > 0, padding)
        padded_tasks = np.pad(self.state.tasks > 0, padding)
        padded_wall = np.pad(np.ones((BOARD_YSIZE, BOARD_XSIZE), dtype=np.bool_), padding, constant_values=0)

        impostor_channel = np.zeros((OBS_YSIZE, OBS_XSIZE), dtype=np.bool_)
        crewmate_channel = np.zeros((OBS_YSIZE, OBS_XSIZE), dtype=np.bool_)

        player_locations = np.array([p.location for p in self.state.players.values()])
        relative_locations = player_locations - np.array([ax, ay]) + np.array([OBS_XSIZE // 2, OBS_YSIZE // 2])
        valid_indices = (relative_locations[:, 0] >= 0) & (relative_locations[:, 0] < OBS_XSIZE) & (relative_locations[:, 1] >= 0) & (relative_locations[:, 1] < OBS_YSIZE)

        impostor_channel[relative_locations[valid_indices, 1], relative_locations[valid_indices, 0]] = np.array([p.impostor for p in self.state.players.values()])[valid_indices]
        crewmate_channel[relative_locations[valid_indices, 1], relative_locations[valid_indices, 0]] = np.array([not p.impostor for p in self.state.players.values()])[valid_indices]

        return np.stack(
            [
                impostor_channel,
                crewmate_channel,
                padded_dead[ay:ay + OBS_YSIZE, ax:ax + OBS_XSIZE],
                padded_tasks[ay:ay + OBS_YSIZE, ax:ax + OBS_XSIZE],
                padded_wall[ay:ay + OBS_YSIZE, ax:ax + OBS_XSIZE],
            ]
        )



    def legal_mask(self, agent: AgentID) -> np.ndarray[int, np.dtype[np.bool_]]:
        """Returns a mask that indicates which actions are legal for the given agent."""
        player_x, player_y = self.state.players[agent].location

        # forbid moving out of bounds
        out_of_bounds = (
            player_x == 0,
            player_x == BOARD_XSIZE - 1,
            player_y == 0,
            player_y == BOARD_YSIZE - 1,
        )
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[out_of_bounds] = False

        # always allow the WAIT action
        mask[Actions.WAIT] = True

        return mask


    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def observation_space(self, _: AgentID):
        return spaces.MultiBinary([OBS_NUM_CHANNELS, OBS_YSIZE, OBS_YSIZE])

    @functools.lru_cache(maxsize=None)
    def action_space(self, _: AgentID):
        return spaces.Discrete(ACTION_SPACE_SIZE)

    def step(
        self, actions: dict[AgentID, Action]
    ) -> tuple[
        # Observations
        dict[AgentID, Observation],
        # Reward
        dict[AgentID, float],
        # Terminated (do not use this agent again, an error will occur)
        dict[AgentID, bool],
        # Truncated (do not use this agent again, an error will occur)
        dict[AgentID, bool],
        # Info (nothing rn)
        dict[str, dict],
    ]:
        # move all players according to their moves (not assuming they checked legal moves)
        for agent, action in actions.items():
            agent_x, agent_y = self.state.players[agent].location
            match action:
                case Actions.MOVE_LEFT:
                    if agent_x != 0:
                        agent_x -= 1
                case Actions.MOVE_RIGHT:
                    if agent_x != BOARD_XSIZE - 1:
                        agent_x += 1
                case Actions.MOVE_UP:
                    if agent_y != 0:
                        agent_y -= 1
                case Actions.MOVE_DOWN:
                    if agent_y != BOARD_YSIZE - 1:
                        agent_y += 1
                case Actions.WAIT:
                    pass
            self.state.players[agent].location = (agent_x, agent_y)

        impostor_locs = np.zeros((BOARD_YSIZE, BOARD_XSIZE), dtype=int)
        crewmate_locs = np.zeros((BOARD_YSIZE, BOARD_XSIZE), dtype=int)

        # count impostors and crewmates in each location
        for p in self.state.players.values():
            x, y = p.location
            if p.impostor:
                impostor_locs[y, x] += 1
            else:
                crewmate_locs[y, x] += 1
        observations = {k: self.observe(k) for k in self.state.players.keys()}
        rewards = {k: 0.0 for k in self.state.players.keys()}
        terminateds = {k: False for k in self.state.players.keys()}
        infos = {k: {} for k in self.state.players.keys()}

        # if impostor, we get a reward for each crewmate in the same location
        for k, p in self.state.players.items():
            if p.impostor:
                rewards[k] += crewmate_locs[p.location[1], p.location[0]]

        # if crewmate, we get a reward if we are on the task location
        # if crewmate, we get a penalty and die if we are on the same location as an impostor
        for k, p in self.state.players.items():
            if not p.impostor:
                # award 0.5 for each task in the same location
                # remove one of the tasks in the same location
                if self.state.tasks[p.location[1], p.location[0]] > 0:
                    rewards[k] += 0.5
                    self.state.tasks[p.location[1], p.location[0]] -= 1

                if impostor_locs[p.location[1], p.location[0]] > 0:
                    rewards[k] -= 0.5
                    self.state.dead[p.location[1], p.location[0]] += 1
                    terminateds[k] = True

        # increment step counter
        self.steps += 1
        if self.steps >= MAX_STEPS:
            truncateds = {k: True for k in self.state.players.keys()}
        else:
            truncateds = {k: False for k in self.state.players.keys()}

        # remove dead players
        self.state.players = {
            k: p for k, p in self.state.players.items() if not (terminateds[k] or truncateds[k])
        }
        self.agents = list(self.state.players.keys())

        return observations, rewards, terminateds, truncateds, infos

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> dict[AgentID, Observation]:
        self.state = self.initial_state
        self.agents = list(self.state.players.keys())
        self.steps = 0
        return {k: self.observe(k) for k in self.state.players.keys()}