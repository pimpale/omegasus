import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID
import gymnasium as gym
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


def print_action(action: Action):
    if action == Actions.MOVE_LEFT:
        print("Move Left")
    elif action == Actions.MOVE_RIGHT:
        print("Move Right")
    elif action == Actions.MOVE_UP:
        print("Move Up")
    elif action == Actions.MOVE_DOWN:
        print("Move Down")
    elif action == Actions.WAIT:
        print("Wait")
    else:
        print("Unknown Action")


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

        # pad the dead and tasks channel and extract an OBS_YSIZE x OBS_XSIZE window around the player
        dead_channel = np.pad(
            self.state.dead > 0,
            padding,
            mode="constant",
            constant_values=False,
        )[ay : ay + OBS_YSIZE, ax : ax + OBS_XSIZE]

        task_channel = np.pad(
            self.state.tasks > 0,
            padding,
            mode="constant",
            constant_values=False,
        )[ay : ay + OBS_YSIZE, ax : ax + OBS_XSIZE]

        wall_channel = np.ones((OBS_YSIZE, BOARD_XSIZE), dtype=np.bool_)
        wall_channel[ay : ay + OBS_YSIZE, ax : ax + OBS_XSIZE] = False

        impostor_channel = np.full((OBS_YSIZE, OBS_XSIZE), False, dtype=np.bool_)
        crewmate_channel = np.full((OBS_YSIZE, OBS_XSIZE), False, dtype=np.bool_)
        impostor_channel[self.state.players.values()] = True
        crewmate_channel[self.state.players.values() ^ self.state.impostors] = True

        # cast rays from the player's location in all directions
        rays = []
        for i in range(360):
            angle = i * np.pi / 180
            dx = np.cos(angle)
            dy = np.sin(angle)
            rays.append((dx, dy))

        # for each ray, check if it hits a wall or a player
        for ray in rays:
            x, y = ax, ay
            while x >= 0 and x < BOARD_XSIZE and y >= 0 and y < BOARD_YSIZE:
                if self.state.walls[y][x]:
                    break
                elif self.state.players[y][x]:
                    if self.state.players[y][x].impostor:
                        impostor_channel[y - ay][x - ax] = True
                    else:
                        crewmate_channel[y - ay][x - ax] = True
                    break
                x += dx
                y += dy

        return np.stack(
            [
                impostor_channel,
                crewmate_channel,
                dead_channel,
                task_channel,
                wall_channel,
            ]
        ).astype(np.bool_)


    def legal_mask(self, agent: AgentID) -> np.ndarray[int, np.dtype[np.bool_]]:

        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.bool_)
        player_x, player_y = self.state.players[agent].location

        # forbid moving out of bounds
        mask[Actions.MOVE_LEFT] = player_x > 0
        mask[Actions.MOVE_RIGHT] = player_x < BOARD_XSIZE - 1
        mask[Actions.MOVE_UP] = player_y > 0
        mask[Actions.MOVE_DOWN] = player_y < BOARD_YSIZE - 1

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
        # Reward
        dict[AgentID, float],
        # Truncated (do not use this agent again, an error will occur)
        dict[AgentID, bool],
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

        # count impostors and crewmates in each location
        impostor_locs = np.zeros((BOARD_YSIZE, BOARD_XSIZE))
        crewmate_locs = np.zeros((BOARD_YSIZE, BOARD_XSIZE))
        for p in self.state.players.values():
            x, y = p.location
            if p.impostor:
                impostor_locs[y, x] += 1
            else:
                crewmate_locs[y, x] += 1

        # update rewards
        rewards = {k: 0.0 for k in self.state.players.keys()}
        for k, p in self.state.players.items():
            if p.impostor:
                rewards[k] += crewmate_locs[p.location]
            else:
                if self.state.tasks[p.location] > 0:
                    rewards[k] += 0.5
                    self.state.tasks[p.location] -= 1
                if impostor_locs[p.location] > 0:
                    rewards[k] -= 0.5
                    self.state.dead[p.location] += 1

        # increment step counter
        self.steps += 1
        if self.steps >= MAX_STEPS:
            truncateds = {k: True for k in self.state.players.keys()}
        else:
            truncateds = {k: False for k in self.state.players.keys()}

        # remove dead players
        self.state.players = {
            k: p for k, p in self.state.players.items() if not (truncateds[k])
        }
        self.agents = list(self.state.players.keys())

        return rewards, truncateds


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
