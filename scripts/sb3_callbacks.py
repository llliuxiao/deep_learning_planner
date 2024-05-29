import os
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class CustomCallback(BaseCallback):
    def __init__(self, verbose=1, check_freq=1024, log_dir=None):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = os.path.join(log_dir, "best_model")
        self.max_reward = -math.inf
        self.max_reward_step = 0
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        infos = self.locals.get('infos')
        if infos is not None:
            self.logger.record_mean('Reward/arrival_rew', np.mean([info['arrival'] for info in infos]))
            self.logger.record_mean('Reward/collision_rew', np.mean([info['collision'] for info in infos]))
            self.logger.record_mean('Reward/angular_rew', np.mean([info['angular'] for info in infos]))
            self.logger.record_mean("Reward/direction_rew", np.mean([info["direction"] for info in infos]))
            self.logger.record_mean("Reward/total_rew", np.mean([info["reward"] for info in infos]))
        if self.n_calls % self.check_freq == 0:
            value = self.logger.name_to_value["Reward/total_rew"]
            if value >= self.max_reward:
                self.model.save(self.log_dir)
                self.max_reward_step = self.n_calls
                print(f"Last mean reward per step: {value:.2f}, Reach the best!")
                print(f"save best model(per step) on step{self.n_calls}")
                self.max_reward = value
            else:
                print(f"Last mean reward per step: {value:.2f}")
                print(f"save best model(per step) on step{self.max_reward_step}")
        return True

    def _on_training_end(self) -> None:
        print(f"save best model(per step) on step{self.max_reward_step}")


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.max_reward_step = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_end(self) -> None:
        print(f"self best model(per episode) on step {self.max_reward_step}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.max_reward_step = self.n_calls
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True
