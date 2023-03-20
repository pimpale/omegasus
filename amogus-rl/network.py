import env

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from env import ACTION_SPACE_SIZE, BOARD_XSIZE, BOARD_YSIZE, NUM_CHANNELS
# Hyperparameters
BOARD_CONV_FILTERS = 10

ACTOR_LR = 1e-4  # Lower lr stabilises training greatly
CRITIC_LR = 5e-4  # Lower lr stabilises training greatly
GAMMA = 0.80
PPO_EPS = 0.1
PPO_EPOCHS = 20

# Converts an observation into a numpy array with dims (Channel, Width, Height)
def reshape_board(obs: env.Observation) -> np.ndarray:  
    channels = [obs.crewmate_channel, obs.impostor_channel, obs.task_channel, obs.self_channel]
    return np.stack(channels).astype(np.float32)

# output in (Batch, Channel, Width, Height)
def obs_batch_to_tensor(o_batch: list[env.Observation], device: torch.device) -> torch.Tensor:
    # Convert state batch into correct format
    return torch.from_numpy(np.stack([reshape_board(o) for o in o_batch])).to(device)

# output in (Batch, Channel, Width, Height)
def obs_to_tensor(o: env.Observation, device: torch.device) -> torch.Tensor:
    # we need to add a batch axis and then convert into a tensor
    return torch.from_numpy(np.stack([reshape_board(o)])).to(device)


def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=NUM_CHANNELS, out_channels=BOARD_CONV_FILTERS, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(BOARD_XSIZE*BOARD_YSIZE*BOARD_CONV_FILTERS, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Width, Height)
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        # flatten everything except for batch
        x = torch.flatten(x, 1)
        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # delete extra dimension
        # output in (Batch,)
        output = x.view((x.shape[0]))
        return output


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.action_size = ACTION_SPACE_SIZE

        self.conv1 = nn.Conv2d(
            in_channels=NUM_CHANNELS, out_channels=BOARD_CONV_FILTERS, kernel_size=3, padding='same')
        self.fc1 = nn.Linear(BOARD_XSIZE*BOARD_YSIZE*BOARD_CONV_FILTERS, 256)
        self.fc2 = nn.Linear(256, ACTION_SPACE_SIZE)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast to float32
        # x in (Batch, Channels, Width, Height)
        x = x.to(torch.float32)
        # apply convolutions
        x = self.conv1(x)
        x = F.relu(x)
        # flatten everything except for batch
        x = torch.flatten(x, 1)
        # apply fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output in (Batch, Width)
        output = F.softmax(x, dim=1)
        return output

# https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations
# The standard policy gradient is given by:
# $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)A^{\pi_{\theta}}(s_t, a_t)$
# where:
# * $\pi_{\theta}(a_t|s_t)$ is the current policy's probability to perform action $a_t$ given $s_t$
# * $A^{\pi_{\theta}}(s_t, a_t)$ is the current value network's guess of the advantage of action $a_t$ at $s_t$
def compute_ppo_loss(
    # Old policy network's probability of choosing an action
    # in (Batch, Action)
    pi_thetak_given_st: torch.Tensor,
    # Current policy network's probability of choosing an action
    # in (Batch, Action)
    pi_theta_given_st: torch.Tensor,
    # One hot encoding of which action was chosen
    # in (Batch, Action)
    a_t: torch.Tensor,
    # Advantage of the chosen action
    A_pi_theta_given_st_at: torch.Tensor,
) -> torch.Tensor:
    # in (Batch,)
    pi_theta_given_st_at = torch.sum(pi_theta_given_st*a_t, 1)
    pi_thetak_given_st_at = torch.sum(pi_thetak_given_st*a_t, 1)

    # the likelihood ratio (used to penalize divergence from the old policy)
    likelihood_ratio = pi_theta_given_st_at / pi_thetak_given_st_at

    # in (Batch,)
    ppo2loss_at_t = torch.minimum(
        likelihood_ratio*A_pi_theta_given_st_at,
        torch.clip(likelihood_ratio, 1-PPO_EPS, 1+PPO_EPS) *
        A_pi_theta_given_st_at
    )

    # in (Batch,)
    entropy_at_t = -torch.sum(torch.log(pi_theta_given_st)
                              * pi_theta_given_st, 1)

    total_loss_at_t = -ppo2loss_at_t - 0.25*entropy_at_t

    # we take the average loss over all examples
    return total_loss_at_t.mean()


def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[env.Observation],
    action_batch: list[env.Action],
    oldpolicy_batch: list[np.ndarray],
    advantage_batch: list[env.Advantage],
    value_batch: list[env.Value],
) -> tuple[list[float], list[float]]:
    # assert that the models are on the same device
    assert next(critic.parameters()).device == next(actor.parameters()).device
    # assert that the batch_lengths are the same
    assert len(observation_batch) == len(action_batch)
    assert len(observation_batch) == len(oldpolicy_batch)
    assert len(observation_batch) == len(advantage_batch)
    assert len(observation_batch) == len(value_batch)

    # get device
    device = next(critic.parameters()).device

    # convert data to tensors on correct device

    # in (Batch, Width, Height)
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)

    # in (Batch,)
    true_value_batch_tensor = torch.tensor(
        value_batch, dtype=torch.float32, device=device)

    # in (Batch, Action)
    chosen_action_tensor = F.one_hot(torch.tensor(action_batch).to(
        device).long(), num_classes=ACTION_SPACE_SIZE) 

    # in (Batch, Action)
    old_policy_action_probs_batch_tensor = torch.tensor(
        np.array(oldpolicy_batch)).to(device)

    # in (Batch,)
    advantage_batch_tensor = torch.tensor(advantage_batch).to(device)


    # train actor
    actor_losses:list[float] = []
    for _ in range(PPO_EPOCHS):
        actor_optimizer.zero_grad()
        current_policy_action_probs = actor.forward(observation_batch_tensor)
        actor_loss = compute_ppo_loss(old_policy_action_probs_batch_tensor,
                                      current_policy_action_probs, chosen_action_tensor, advantage_batch_tensor)
        actor_loss.backward()
        actor_optimizer.step()
        actor_losses.append(actor_loss.item())

    # train critic
    critic_optimizer.zero_grad()
    pred_value_batch_tensor = critic.forward(observation_batch_tensor)
    critic_loss = F.mse_loss(pred_value_batch_tensor, true_value_batch_tensor) # TODO: look into other losses 
    critic_loss.backward()
    critic_optimizer.step()

    # return the respective losses
    return (actor_losses, [float(critic_loss)]*PPO_EPOCHS)


# computes advantage using Generalized Advantage Estimation 
def compute_advantage(
    critic: Critic,
    trajectory_observations: list[env.Observation],
    trajectory_rewards: list[env.Reward],
) -> list[env.Advantage]:
    trajectory_len = len(trajectory_rewards)

    assert len(trajectory_observations) == trajectory_len
    assert len(trajectory_rewards) == trajectory_len

    trajectory_advantages = np.zeros(trajectory_len)

    # calculate the value of the state at the end
    last_obs = obs_to_tensor(
        trajectory_observations[-1], next(critic.parameters()).device)
    last_obs_value = critic.forward(last_obs)[0]

    trajectory_advantages[-1] = last_obs_value + trajectory_rewards[-1]

    # Use GAMMA to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        trajectory_advantages[t] = trajectory_rewards[t] + \
            GAMMA * trajectory_advantages[t + 1]

    return list(trajectory_advantages)

# computes what the critic network should have predicted
# return over trajectory at each time step 
def compute_value(
    trajectory_rewards: list[env.Reward],
) -> list[env.Value]:
    trajectory_len = len(trajectory_rewards)

    v_batch = np.zeros(trajectory_len)

    v_batch[-1] = trajectory_rewards[-1]

    # Use GAMMA to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + GAMMA * v_batch[t + 1]

    return list(v_batch)
