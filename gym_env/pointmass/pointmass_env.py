import gym
import numpy as np
from gym_env.pointmass.agent import Agent
import random


class PointMassEnv(gym.Env):
    def __init__(self, args, xlim=3, radius=1):
        # super(PointMassEnv, self).__init__(log=log, args=args)
        # xlim = 10, radius = 5
        self.xlim = xlim
        self.radius = radius
        self.dim = 2
        self.continuous_action = True
        self.max_acc = 2

        self.observation_shape = (4,)
        self.action_space = (5,) if not self.continuous_action else (2,)
        self.observation_space = gym.spaces.Box(low=-1e7, high=1e7, shape=self.observation_shape)
        self.action_space = gym.spaces.Box(low=-5, high=5, shape=self.action_space)
        self.constraint_maximum = 0
        self.action_minimum = -1
        self.action_maximum = 1
        self.dt = .01

        # self.pad = np.max(self.observation_shape) - 2
        # self.height = self.observation_shape[0]
        # self.half_width = int(self.observation_shape[1] / 2)
        #
        # self.base_gridmap_array = self._load_gridmap_array()
        # self.base_gridmap_image = self._to_image(self.base_gridmap_array)
        #
        # self.agents = []
        # for i_agent, agent_type in enumerate(["prey"] + ["predator" for _ in range(self.args.n_predator)]):
        #     agent = Agent(i_agent, agent_type, self.base_gridmap_array)
        #     self.agents.append(agent)
        self.agent = Agent()
        self.agent.color = np.array([0.35, 0.35, 0.85])
        self.agent.position = None
        self.agent.orientation = None

        self.landmark = Agent()
        self.landmark.color = np.array([0, 0, 0])
        self.landmark.position = None
        self.landmark.orientation = None

    def _reset_agent(self):
        self.agent.position = 0.1 * self.xlim * np.random.uniform(0, 1, self.dim) - self.xlim
        # self.landmark.position = np.array([9, 9])
        self.landmark.position = self.xlim - 0.1 * self.xlim * np.random.uniform(0, 1, self.dim)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self._reset_agent()
        # self.agent.position = 0.1 * self.xlim * np.random.uniform(0, 1, self.dim) - self.xlim
        # # self.landmark.position = np.array([9, 9])
        # self.landmark.position = self.xlim - 0.1 * self.xlim * np.random.uniform(0, 1, self.dim)
        # self.agent.orientation = np.random.uniform(0, 2 * np.pi, 1)
        observations = self._get_obs()
        return observations

    def _get_obs(self):
        # env_info = np.array([self.xlim, self.radius])
        # observations = np.concatenate((self.agent.position, self.landmark.position, env_info))
        observations = np.concatenate((self.agent.position, self.landmark.position))
        return observations

    def step(self, action):
        pos = self.agent.position
        forward_len = 0.5
        if not self.continuous_action:
            if action == 0:
                pos[0] += forward_len
            elif action == 1:
                pos[0] -= forward_len
            elif action == 2:
                pos[1] += forward_len
            elif action == 3:
                pos[1] -= forward_len
            elif action == 4:
                pass
            else:
                raise ValueError
        else:
            action = np.clip(action, -self.max_acc, self.max_acc)
            if len(action.shape) == 2:
                action = action.flatten()
            pos += action * self.dt

        self.agent.position = pos
        next_obs = self._get_obs()

        reward = -np.linalg.norm(self.agent.position - self.landmark.position)
        # return Step(next_obs, reward, False)

        cost = 0 if np.linalg.norm(self.agent.position) > self.radius and \
            np.max(np.abs(self.agent.position)) < self.xlim else 1
        info = {}
        info['cost'] = cost

        # cost = 0 if np.linalg.norm(self.agent.position) > self.radius else 1

        # d1 = self.radius - np.linalg.norm(self.agent.position)
        # d2 = np.max(np.abs(self.agent.position)) - self.xlim
        # alpha1 = 1 if d1 > 0 else 0.5
        # alpha2 = 1 if d2 > 0 else 0.5
        # cost = alpha1*d1 + alpha2*d2
        # cost = alpha1 * d1
        # cost = alpha2 * d2

        # cost = 1 / (1 + np.exp(-.1*d1))

        # cost = self.radius - np.linalg.norm(self.agent.position) + \
        #        np.max(np.abs(self.agent.position)) - self.xlim

        done = True if np.linalg.norm(self.agent.position - self.landmark.position) <= 1e-1 else False

        return next_obs, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError
