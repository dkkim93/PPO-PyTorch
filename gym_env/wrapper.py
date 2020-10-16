import gym
import numpy as np


class PendulumCostWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(PendulumCostWrapper, self).__init__(env)

    def step(self, u):
        th, thdot = self.state

        g = self.g
        m = 1.  # Default value from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        l = 1.  # Default value from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111
        self.state = np.array([newth, newthdot])

        bound = 4.5
        constraint = 0.06 if np.linalg.norm(self.state[1]) >= bound else 0
        info = {'cost': constraint}
        return self._get_obs(), -costs[0], False, info

    def reset(self):
        hign_max = 2.
        high = np.array([np.pi, 1])
        high = high / np.linalg.norm(high) * hign_max
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot])
        if len(obs.shape) == 2:
            obs = obs.flatten()
        return obs


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
