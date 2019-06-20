import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR, A2C
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np


best_mean_reward, n_steps = -np.inf, 0
log_dir = "logs/"

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward


    # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True

env = Monitor(gym.make('gym_threes:threes-v0'), log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=100000, callback=callback)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()