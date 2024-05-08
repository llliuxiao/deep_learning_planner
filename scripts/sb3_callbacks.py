import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True


class RewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardCallback, self).__init__(verbose)

    def _on_step(self):
        infos = self.locals.get('infos')
        if infos is not None:
            self.logger.record_mean('Train/arrival_rew', np.mean([info['arrival'] for info in infos]))
            self.logger.record_mean('Train/collision_rew', np.mean([info['collision'] for info in infos]))
            self.logger.record_mean('Train/angular_rew', np.mean([info['angular'] for info in infos]))
        return True

# class RewardCallback(BaseCallback):
#     def __init__(self, verbose=1, interval=256):
#         super(RewardCallback, self).__init__(verbose)
#         self.arrival_rew = []
#         self.collision_rew = []
#         self.angular_rew = []
#         self.interval = interval
#
#     def _on_step(self):
#         infos = self.locals.get('infos')
#         self.arrival_rew.append(np.mean([info['arrival'] for info in infos]))
#         self.collision_rew.append(np.mean([info['collision'] for info in infos]))
#         self.angular_rew.append(np.mean([info['angular'] for info in infos]))
#         if len(self.angular_rew) == self.interval:
#             self.logger.record('Train/arrival_rew', np.mean(self.arrival_rew))
#             self.logger.record('Train/collision_rew', np.mean(self.collision_rew))
#             self.logger.record('Train/angular_rew', np.mean(self.angular_rew))
#             self.angular_rew.clear()
#             self.collision_rew.clear()
#             self.arrival_rew.clear()
#         return True

