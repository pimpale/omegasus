import numpy as np
from dataclasses import dataclass
from typing import Any, TypeAlias, Literal
import copy

BOARD_XSIZE = 5
BOARD_YSIZE = 5
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
    dead: bool


@dataclass
class State:
    """Overall state of the game"""
    players: list[PlayerState]
    tasks: np.ndarray[Any, np.dtype[np.bool8]]


@dataclass
class Observation:  # note, not differentiating between player and impostor
    """Observation by a single player of the game"""
    player_channel: np.ndarray[Any, np.dtype[np.int8]]
    task_channel: np.ndarray[Any, np.dtype[np.bool8]]
    self_channel: np.ndarray[Any, np.dtype[np.int8]]
    impostor_channel: np.ndarray[Any, np.dtype[np.int8]]
    dead_channel: np.ndarray[Any, np.dtype[np.int8]]

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

def print_obs(obs: Observation):
    for y in range(BOARD_YSIZE):
        q = ''
        for x in range(BOARD_XSIZE):
            c = '⬛'

            if obs.dead_channel[y, x]:
                c = '💀'
            elif obs.impostor_channel[y, x]:
                if obs.self_channel[y, x]:
                    c = '😈'
                else:
                    c = '👽'
            elif obs.player_channel[y, x]:
                if obs.self_channel[y, x]:
                    c = '😇'
                else:
                    c = '🧑‍🚀'
    
            if obs.task_channel[y, x]:
                if c == '⬛':
                    c = '📦'
            q += c
        print(q)


def initial_state() -> State:
    task_locations = np.zeros((BOARD_XSIZE, BOARD_YSIZE), dtype=np.bool8)
    task_locations[2, 2] = True
    return State(
        # Players
        [
            # Impostor at (1, 1)
            PlayerState((1, 1), True, False),
            # Crewmate at (3, 3)
            PlayerState((3, 3), False, False)
        ],
        # Tasks
        task_locations,
    )


class Env():
    def __init__(self):
        self.steps = 0
        self.state: State = initial_state()

    def reset(self) -> None:
        self._game_over = False
        self.state = initial_state()

    def observe(self, player: Player) -> Observation:
        """Observation by a single player of the game"""
        player_channel = np.zeros((BOARD_XSIZE, BOARD_YSIZE), dtype=np.int8)
        impostor_channel = np.zeros((BOARD_XSIZE, BOARD_YSIZE), dtype=np.int8)
        self_channel = np.zeros((BOARD_XSIZE, BOARD_YSIZE), dtype=np.int8)
        dead_channel = np.zeros((BOARD_XSIZE, BOARD_YSIZE), dtype=np.int8)
        task_channel = self.state.tasks

        for ip, p in enumerate(self.state.players):
            x, y = p.location
            if ip == player:
                self_channel[y, x] = 1
            if p.dead:
                dead_channel[y, x] = 1
            else:
                player_channel[y, x] = 1
                if p.impostor:
                    impostor_channel[y, x] = 1

        return Observation(
            player_channel,
            task_channel,
            self_channel,
            impostor_channel,
            dead_channel,
        )
    
    def game_over(self) -> bool:
        return self.steps >= 100

    def game_over_for(self, player: Player) -> bool:
        return self.state.players[player].dead or self.game_over()

    def legal_mask(self, player: Player) -> np.ndarray[Any, np.dtype[np.bool8]]:
        mask = np.ones(ACTION_SPACE_SIZE)
        player_x, player_y = self.state.players[player].location

        # forbid moving out of bounds
        if player_x == 0:
            mask[Actions.MOVE_LEFT] = 0
        if player_x == BOARD_XSIZE - 1:
            mask[Actions.MOVE_RIGHT] = 0
        if player_y == 0:
            mask[Actions.MOVE_UP] = 0
        if player_y == BOARD_YSIZE - 1:
            mask[Actions.MOVE_DOWN] = 0

        mask[Actions.WAIT] = 1

        return mask.astype(np.bool8)

    # play an action (since this is a multi-agent game, we'll only know the rewards at the end of the step)
    def play(self, action: Action, player: Player) -> None:
        # assert player is live
        assert not self.state.players[player].dead

        # make sure the move is legal
        assert (self.legal_mask(player)[action] == 1)

        # change location of the player
        player_x, player_y = self.state.players[player].location
        match action:
            case Actions.MOVE_LEFT:
                player_x -= 1
            case Actions.MOVE_RIGHT:
                player_x += 1
            case Actions.MOVE_UP:
                player_y -= 1
            case Actions.MOVE_DOWN:
                player_y += 1
            case Actions.WAIT:
                pass
            case _:
                raise ValueError("Invalid action")
        
        # move player
        self.state.players[player].location = (player_x, player_y)

    
    # we pay out rewards when we step
    def step(self)-> np.ndarray[Any, np.dtype[Reward]]:
        impostor_locs = np.zeros((BOARD_XSIZE, BOARD_YSIZE))
        crewmate_locs = np.zeros((BOARD_XSIZE, BOARD_YSIZE))

        # count impostors and crewmates in each location
        for p in self.state.players:
            if p.dead:
                continue
            x, y = p.location
            if p.impostor:
                impostor_locs[y, x] += 1
            else:
                crewmate_locs[y, x] += 1

        rewards = np.zeros(len(self.state.players), dtype=Reward)

        # if impostor, we get a reward for each crewmate in the same location
        for i, p in enumerate(self.state.players):
            if p.impostor:
                x, y = p.location
                rewards[i] += crewmate_locs[y, x]
        
        # if crewmate, we get a reward if we are on the task location
        # we get a penalty and die if we are on the same location as an impostor
        for i, p in enumerate(self.state.players):
            if (not p.impostor) and (not p.dead):
                x, y = p.location
                rewards[i] += self.state.tasks[y, x]
                if impostor_locs[y, x] > 0:
                    rewards[i] -= 10
                    self.state.players[i].dead = True

        self.steps += 1
        return rewards