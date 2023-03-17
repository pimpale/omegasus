import numpy as np
from dataclasses import dataclass
from typing import Any, TypeAlias, Literal

BOARD_XSIZE = 3
BOARD_YSIZE = 3
ACTION_SPACE_SIZE = 5  # U, D, L, R, W
NUM_CHANNELS = 4  # players, imposters, tasks, self

# integer [0, 1, 2, 3, 4] <=> [U, D, L, R, W]
Action: TypeAlias = np.int8

# Reward for the agent
Reward: TypeAlias = np.float32

# Value of an observation for an agent
Value: TypeAlias = np.float32

# Advantage of a particular action for an agent
Advantage: TypeAlias = np.float32

# The identity of the player
Player: TypeAlias = np.int8

# Action IDs


class Actions:
    MV_LEFT = Action(0)
    MV_RIGHT = Action(1)
    MV_UP = Action(2)
    MV_DOWN = Action(3)
    WAIT = Action(4)


@dataclass
class PlayerState:
    """State of a single player"""
    location: tuple[int, int]
    impostor: bool
    dead: bool


@dataclass
class State:
    """Overall state of the game"""
    players: list[PlayerState]
    task_locations: list[tuple[int, int]]


@dataclass
class Observation:  # note, not differentiating between player and impostor
    """Observation by a single player of the game"""
    players: list[PlayerState]
    task_locations: list[tuple[int, int]]
    self_id: Player


def print_obs(obs: Observation):
    print("- "*BOARD_XSIZE)
    for i in range(BOARD_XSIZE):
        q = '|'
        for j in range(BOARD_YSIZE):
            c = ' '
            for p in obs.players:
                if p.location == (i, j):
                    if p.impostor:
                        c = '👽'
                    else:
                        c = '😀'
    
            if (i, j) in obs.task_locations:
                c = '📦'
            
            q += c
        q += ' |'
        print(q)
    print("- "*BOARD_XSIZE)
    print()


def initial_state() -> State:
    return State(
        # Players
        [
            # Impostor at (1, 1)
            PlayerState((1, 1), True, False),
            # Crewmate at (3, 3)
            PlayerState((3, 3), False, False)
        ],
        # Task at (2, 2)
        [(2, 2)],
    )


class Env():
    def __init__(self):
        self.steps = 0
        self.state: State = initial_state()

    def reset(self) -> None:
        self._game_over = False
        self.state = initial_state()

    def observe(self, player: Player) -> Observation:
        return Observation(self.state.players, self.state.task_locations, player)

    def game_over(self) -> bool:
        return self.steps >= 100

    def game_over_for(self, player: Player) -> bool:
        return self.state.players[player].dead or self.game_over()

    def legal_mask(self, player: Player) -> np.ndarray[Any, np.dtype[np.bool8]]:
        mask = np.ones(ACTION_SPACE_SIZE)
        player_x, player_y = self.state.players[player].location

        # forbid moving out of bounds
        if player_x == 0:
            mask[Actions.MV_LEFT] = 0
        if player_x == BOARD_XSIZE - 1:
            mask[Actions.MV_RIGHT] = 0
        if player_y == 0:
            mask[Actions.MV_UP] = 0
        if player_y == BOARD_YSIZE - 1:
            mask[Actions.MV_DOWN] = 0

        mask[Actions.WAIT] = 1

        return mask.astype(np.bool8)

    def step(self, action: Action, player: Player) -> Reward:
        self.steps += 1
        # assert player is live
        assert not self.state.players[player].dead

        # make sure the move is legal
        assert (self.legal_mask(player)[action] == 1)

        # change location of the player
        player_x, player_y = self.state.players[player].location
        match action:
            case Actions.MV_LEFT:
                player_x -= 1
            case Actions.MV_RIGHT:
                player_x += 1
            case Actions.MV_UP:
                player_y -= 1
            case Actions.MV_DOWN:
                player_y += 1
            case Actions.WAIT:
                pass
            case _:
                raise ValueError("Invalid action")

        reward = 0

        # if impostor, kill all crewmates in the same location
        if self.state.players[player].impostor:
            for p in self.state.players:
                if not p.impostor and not p.dead and p.location == (player_x, player_y):
                    p.dead = True
                    reward += 1

        # if crewmate, kill self if in same location as impostor
        if not self.state.players[player].impostor:
            impostor = next(filter(lambda x: x.impostor, self.state.players))
            if impostor.location == (player_x, player_y):
                self.state.players[player].dead = True
                reward -= 1

        # if crewmate, get reward if in same location as a task
        if not self.state.players[player].impostor:
            if (player_x, player_y) in self.state.task_locations:
                reward += 1

        return Reward(reward)
