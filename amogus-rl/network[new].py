import ray

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import env

from ray import tune
from ray.rllib.models import ModelCatalog, TorchModelV2
from ray.rllib.policy.zsample_batch import SampleBatch
from ray.rllib.policy import Policy, register_policy
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.annotations import override
from ray.rllib.evaluation import postprocessing
from ray.rllib.models.preprocessors import Preprocessor
import yaml



# Initialize Ray
ray.init()

# Load the configuration from the YAML file
config = yaml.safe_load(open("config.yaml"))

class Actor(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, custom_model, name, custom_model_config):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, custom_model, name)
        nn.Module.__init__(self)
        self.board_width = custom_model_config.get("board_width")
        self.board_height = custom_model_config.get("board_height")
        self.board_conv_filters = custom_model_config.get("board_conv_filters")

        self.conv1 = nn.Conv2d(2, self.board_conv_filters, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            self.board_conv_filters, self.board_conv_filters, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear((self.board_width - 4) * (self.board_height - 4) * self.board_conv_filters, 512)
        self.fc2 = nn.Linear(512, self.board_width)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

class Critic(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, custom_model_2, name, custom_model_config_2):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, custom_model_2, name)
        nn.Module.__init__(self)
        self.board_width = custom_model_2.get("board_width")
        self.board_height = custom_model_2.get("board_height")
        self.board_conv_filters = custom_model_2.get("board_conv_filters")

        self.conv1 = nn.Conv2d(2, self.board_conv_filters, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(
            self.board_conv_filters, self.board_conv_filters, kernel_size=3, padding=0
        )
        self.fc1 = nn.Linear((self.board_width - 4) * (self.board_height - 4) * self.board_conv_filters, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x.view((x.shape[0]))
        return output


# Register your custom models with RLlib
ModelCatalog.register_custom_model("actor", Actor)
ModelCatalog.register_custom_model("critic", Critic)

# Import the AmongUsEnv class from env
from env import AmongUsEnv

# Register the environment with RLlib
tune.register_env("amongus_env", lambda config: AmongUsEnv(config))


# Define the postprocessing function using GAE
def postprocess_ppo(policy,
                    sample_batch,
                    other_agent_batches=None,
                    episode=None):
    # Compute the advantage using GAE
    device = deviceof(policy.model.base_model.critic_model)
    gamma = 0.99  # discount factor for rewards
    lam = 0.95  # smoothing factor for advantages
    trajectory_values = policy.model.base_model.critic_model.value_function().detach().cpu().numpy()
    trajectory_deltas = sample_batch[SampleBatch.REWARDS] + gamma * np.append(
        trajectory_values[1:], [0]) - trajectory_values
    trajectory_advantages = np.zeros_like(sample_batch[SampleBatch.REWARDS])
    running_add = 0
    for t in reversed(range(len(sample_batch[SampleBatch.REWARDS]))):
        running_add = (
            running_add * gamma * lam + trajectory_deltas[t])
        trajectory_advantages[t] = running_add
    
    # Add advantages and value targets fields to sample batch
    sample_batch[] = trajectory_advantages
    sample_batch[SampleBatch.VALUE_TARGETS] = trajectory_advantages + trajectory_values
    
    return sample_batch

# Define the loss function for PPO using RLlib's PPOLoss class
def ppo_loss(policy, model, dist_class, train_batch):

    # Get the critic model from the base model
    critic_model = model.base_model.critic_model
    
    # Get the config values from the policy object
    clip_param = policy.config["clip_param"]
    vf_clip_param = policy.config["vf_clip_param"]
    vf_loss_coeff = policy.config["vf_loss_coeff"]
    
    # Instantiate a PPOLoss object with the given parameters
    loss_obj = ppo.PPOLoss(
            dist_class,
            model,
            value_targets=train_batch[postprocessing.VALUE_TARGETS],
            advantages=train_batch[postprocessing.ADVANTAGES],
            actions=train_batch[SampleBatch.ACTIONS],
            prev_logits=train_batch[SampleBatch.ACTION_DIST_INPUTS],
            vf_preds=train_batch[SampleBatch.VF_PREDS],
            cur_kl_coeff=policy.kl_coeff,
            entropy_coeff=policy.entropy_coeff,
            clip_param=clip_param,
            vf_clip_param=vf_clip_param,
            vf_loss_coeff=vf_loss_coeff)

    
    return loss_obj.loss

# Build a custom policy class using RLlib's build_torch_policy helper
CustomPPOTorchPolicy = ppo.build_torch_policy(
                        name="ppo_policy",
                        loss_fn=ppo_loss,
                        postprocess_fn=postprocess_ppo)

# Register your custom policy class with RLlib
register_policy("ppo_policy", CustomPPOTorchPolicy)


#
def obs_batch_to_tensor(
    o_batch: list[env.Observation], device: torch.device
) -> torch.Tensor:
    # Convert state batch into correct format
    return torch.from_numpy(np.stack(o_batch)).to(device)

def obs_to_tensor(o: env.Observation, device: torch.device) -> torch.Tensor:
    # we need to add a batch axis and then convert into a tensor
    return torch.from_numpy(np.stack([o])).to(device)

def deviceof(m: nn.Module) -> torch.device:
    return next(m.parameters()).device

class CustomPreprocessor(Preprocessor):
    def __init__(self, obs_space, options=None):
        super().__init__(obs_space, options)
        # Specify the shape of the observation tensor
        self._shape = (obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])

    def _init_shape(self, obs_space, options):
        # Return the shape of the observation tensor
        return self._shape

    def transform(self, observation):
        # Convert the observation to a tensor using the obs_to_tensor function
        device = deviceof(self.model)
        return obs_to_tensor(observation, device)

ModelCatalog.register_custom_preprocessor("custom_preprocessor", CustomPreprocessor)


# Define your training function
tune.run(PPOTrainer, config=config)


ray.shutdown()


