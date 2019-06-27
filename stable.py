import gym
from stable_baselines.common.policies import MlpPolicy
import stable_baselines.deepq.policies as deepqpolicies
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR, A2C, DQN, ACER
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import gym
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, help='The RL algorithm to use: {', default='DQN')
parser.add_argument('--timesteps', type=int, help='Maximum timesteps', default=50000000)
parser.add_argument('--name', type=str, help='name', default="")
parser.add_argument('--gamma', type=float, help='Discount factor, should be very close to 0', default=0.99999)
parser.add_argument('--verbosity', type=int, help='Verbosity in 0, 1, 2', default=0)
parser.add_argument('--exploration_fraction', type=float, help="(DQN) fraction of entire training period over which the "
                                                               "exploration rate is annealed", default=0.03)
parser.add_argument('--exploration_final_eps', type=float, help="(DQN) final value of random action probability", default=0.02)

args = parser.parse_args()
name = "-alg " + args.alg + " -n " + args.name


best_mean_reward, n_steps = -np.inf, 0
log_dir = "/var/scratch/ekn274/threesai/logs" + os.sep
monitorpath = os.path.join(log_dir, "monitors", name)

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward


    # Print stats every 1000 calls
    if (n_steps + 1) % 2000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(monitorpath), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'models' + os.sep + name + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True


class MonitorEpisodes(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_steps = 0
        self.rendering = False
        self.invalid = 0

    def step(self, action):
        observation, reward, done, infos = super().step(action)
        if reward < 0:
            self.invalid += 1
        else:
            if self.invalid > 8:
                print("Was unable to do a valid move for ", self.invalid, "steps")
            self.invalid = 0
        if (self.total_steps + 1) % 20000 == 0:
            self.rendering = True
        if self.rendering:
            print("Took action", action)
            print("Received reward", reward)
            print("New state is:")
            self.env.render()
            if done:
                self.rendering = False
        self.total_steps += 1
        return observation, reward, done, infos


os.mkdir(monitorpath)

env = MonitorEpisodes(Monitor(gym.make('gym_threes:threes-v0'), monitorpath, allow_early_resets=True))
env = DummyVecEnv([lambda: env])

if args.alg == 'DQN':
    model = DQN(deepqpolicies.LnMlpPolicy, env, verbose=args.verbosity, tensorboard_log=log_dir, gamma=args.gamma,
                exploration_final_eps=args.exploration_final_eps, exploration_fraction=args.exploration_fraction)
else:
    if args.alg == 'PPO':
        clazz = PPO2
    elif args.alg == 'A2C':
        clazz = A2C
    elif args.alg == 'ACKTR':
        clazz = ACKTR
    elif args.alg == "ACER":
        clazz = ACER
    else:
        raise ValueError("Wrong algorithm")
    model = clazz(MlpPolicy, env, verbose=args.verbosity, tensorboard_log=log_dir, gamma=args.gamma)

print("Starting training of", name)

model.learn(total_timesteps=args.timesteps, callback=callback)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()