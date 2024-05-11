# utils
from typing import Callable, Tuple, Optional
import gymnasium.spaces as spaces
import re

# torch
import numpy as np
import torch
import torch as th
import torch.nn.functional as F

# stable-baseline3
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ActorCriticPolicy


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
        self.kl_loss_fn = th.nn.KLDivLoss(log_target=True, reduction="batchmean").to(self.device)
        self.log_std = th.log(th.tensor([0.03, 0.03], requires_grad=False, dtype=torch.float))

    def calculate_kl_loss(self, obs: PyTorchObs, actions: th.Tensor, actions_log_prob: th.Tensor):
        with th.no_grad:
            features = self.extract_features(obs)
            latent_pi = self.mlp_extractor.forward_actor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
        imitation_log_prob = distribution.log_prob(actions)
        kl_loss = self.kl_loss_fn(actions_log_prob, imitation_log_prob)
        return kl_loss


class CustomPPO(PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.imitation_policy = ImitationPolicy(self.observation_space, self.action_space, self.lr_schedule)

    def load_state_dict(self, model_file: str):
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
        kl_losses = []

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
                """
                define kl loss
                """
                kl_loss = self.imitation_policy.calculate_kl_loss(rollout_data.observations, actions, log_prob)
                kl_losses.append(kl_loss)
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
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + kl_loss

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

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
