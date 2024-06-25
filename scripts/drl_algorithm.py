# utils
from typing import Callable
import gymnasium.spaces as spaces
import re
import gymnasium as gym

# torch
import numpy as np
import torch
import torch as th
import torch.nn.functional as F

# stable-baseline3
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


def init_torch_layer_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight)


class ImitationPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.requires_grad_(False)
        self.eval()

    def get_imitation_action(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        action = self.action_net(latent_pi)
        return action


class CustomPPO(PPO):
    def __init__(self, target_value: float, **kwargs):
        super().__init__(**kwargs)
        self.imitation_policy = ImitationPolicy(self.observation_space, self.action_space, self.lr_schedule)
        self.target_value = torch.tensor(target_value, dtype=torch.float, requires_grad=False).to(self.device)
        self.temperature = th.tensor(1.0, dtype=torch.float).to(self.device)
        self.temperature.requires_grad = True
        self.temperature_optimizer = th.optim.Adam(params=[self.temperature], lr=self.lr_schedule(1.0))

    def load_state_dict(self, model_file: str = None):
        if model_file is None:
            self.policy.apply(init_torch_layer_weights)
            self.imitation_policy.apply(init_torch_layer_weights)
            return
        param = th.load(model_file)["model_state_dict"]
        feature_extractor_param = {}
        mlp_extractor_param = {}
        action_net_param = {}
        for k, v in param.items():
            if re.search("^module.policy_net.", k):
                new_k = re.sub("^module.policy_net.[0-9].", "", k)
                action_net_param[new_k] = v
            elif re.search("^module.mlp_extractor_policy.", k):
                new_k = k.replace("module.mlp_extractor_policy.", "")
                mlp_extractor_param[new_k] = v
            else:
                new_k = k.replace("module.", "")
                feature_extractor_param[new_k] = v
        self.policy.features_extractor.load_state_dict(feature_extractor_param)
        self.policy.mlp_extractor.mlp_extractor_actor.load_state_dict(mlp_extractor_param)
        self.policy.action_net.load_state_dict(action_net_param)
        self.imitation_policy.model.features_extractor.load_state_dict(feature_extractor_param)
        self.imitation_policy.model.mlp_extractor.mlp_extractor_actor.load_state_dict(mlp_extractor_param)
        self.imitation_policy.model.action_net.load_state_dict(action_net_param)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        temperatures = [self.temperature.item()]
        regularization_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                """
                notice that the rollout_data return should not calculate gradient
                """
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                ############################################################################################
                """
                add a regularization here to narrow the gap between imitation policy and reinforcement policy
                """
                imitation_action = self.imitation_policy.get_imitation_action(rollout_data.observations).detach()
                _, imitation_log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, imitation_action)
                regularization = -torch.mean(imitation_log_prob)
                regularization_losses.append(regularization.detach().item())
                ############################################################################################

                ############################################################################################
                """
                total loss function
                """
                loss = (policy_loss + self.ent_coef * entropy_loss +
                        self.vf_coef * value_loss + self.temperature.detach() * regularization)
                ############################################################################################

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                ############################################################################################
                """
                optimize the temperature
                """
                self.temperature.requires_grad = True
                temperature_loss = self.temperature * torch.mean(values.detach() - self.target_value)
                self.temperature_optimizer.zero_grad()
                with torch.autograd.detect_anomaly(True):
                    temperature_loss.backward()
                self.temperature_optimizer.step()
                with torch.no_grad():
                    self.temperature = F.softplus(self.temperature)
                    self.temperature = th.clamp(self.temperature, 0.0, 1.0)
                    temperatures.append(self.temperature.item())
                ############################################################################################

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/temperature", np.mean(temperatures))
        self.logger.record("train/regularization", np.mean(regularization_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


def main():
    env = gym.make("LunarLanderContinuous-v2")
    model = CustomPPO(target_value=100, policy="MlpPolicy", env=env, device="cpu", verbose=1)
    model.load_state_dict()
    model.learn(total_timesteps=int(1e7))
    model.save("custom_ppo_lunar")


if __name__ == "__main__":
    print("test custom ppo with continuous environment LunarLanderContinuous-v2 in gymnasium")
    main()
