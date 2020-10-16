import os
import gym
import copy
import random
import numpy as np
from gym_env.pointmass.config import Config
from collections import deque


class Base(gym.Env):
    def __init__(self, log, args):
        super(Base, self).__init__()

        self.log = log
        self.args = args
        self.config = Config()

    def seed(self, value):
        random.seed(value)
        np.random.seed(value)

    def _load_gridmap_array(self):
        # Ref: https://github.com/xinleipan/gym-gridworld/blob/master/gym_gridworld/envs/gridworld_env.py
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "maze.txt")
        with open(path, 'r') as f:
            gridmap = f.readlines()

        gridmap_array = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), gridmap)))
        return gridmap_array

    def _to_image(self, gridmap_array):
        image = np.zeros((gridmap_array.shape[0], gridmap_array.shape[1], 3), dtype=np.float32)

        for row in range(gridmap_array.shape[0]):
            for col in range(gridmap_array.shape[1]):
                grid = gridmap_array[row, col]

                if grid == self.config.grid_dict["empty"]:
                    image[row, col] = self.config.color_dict["empty"]
                elif grid == self.config.grid_dict["wall"]:
                    image[row, col] = self.config.color_dict["wall"]
                elif grid == self.config.grid_dict["prey"]:
                    image[row, col] = self.config.color_dict["prey"]
                elif grid == self.config.grid_dict["predator"]:
                    image[row, col] = self.config.color_dict["predator"]
                elif grid == self.config.grid_dict["orientation"]:
                    image[row, col] = self.config.color_dict["orientation"]
                else:
                    raise ValueError()

        return image

    def _render_gridmap(self, agent):
        gridmap_image = np.copy(self.base_gridmap_image)

        # Render orientation
        for _agent in self.agents:
            orientation_location = _agent.orientation_location
            gridmap_image[orientation_location[0], orientation_location[1]] = self.config.color_dict["orientation"]
            
        # Render location
        for _agent in self.agents:
            location = _agent.location
            if agent.id == _agent.id:
                gridmap_image[location[0], location[1]] = self.config.color_dict["own"]
            else:
                if agent.type == _agent.type:
                    gridmap_image[location[0], location[1]] = self.config.color_dict["teammate"]
                else:
                    gridmap_image[location[0], location[1]] = self.config.color_dict["opponent"]

        # Pad image
        pad_width = ((self.pad, self.pad), (self.pad, self.pad), (0, 0))
        gridmap_image = np.pad(gridmap_image, pad_width, mode="constant")

        return gridmap_image

    def _reset_agents(self):
        for agent in self.agents:
            location = np.array([
                np.random.choice(self.base_gridmap_array.shape[0]), 
                np.random.choice(self.base_gridmap_array.shape[1])])
            grid = self.base_gridmap_array[location[0], location[1]]

            while grid != self.config.grid_dict["empty"]:
                location = np.array([
                    np.random.choice(self.base_gridmap_array.shape[0]), 
                    np.random.choice(self.base_gridmap_array.shape[1])])
                grid = self.base_gridmap_array[location[0], location[1]]

            agent.location = location
            agent.orientation = self.config.orientation_dict["up"]

    def _get_obs(self, agent, gridmap_image):
        row, col = agent.location[0] + self.pad, agent.location[1] + self.pad

        if agent.orientation == self.config.orientation_dict["up"]:
            observation = gridmap_image[
                row - self.height + 1: row + 1, 
                col - self.half_width: col + self.half_width + 1, :]
        elif agent.orientation == self.config.orientation_dict["right"]:
            observation = gridmap_image[
                row - self.half_width: row + self.half_width + 1, 
                col: col + self.height, :]
            observation = np.rot90(observation, k=1)
        elif agent.orientation == self.config.orientation_dict["down"]:
            observation = gridmap_image[
                row: row + self.height, 
                col - self.half_width: col + self.half_width + 1, :]
            observation = np.rot90(observation, k=2)
        elif agent.orientation == self.config.orientation_dict["left"]:
            observation = gridmap_image[
                row - self.half_width: row + self.half_width + 1, 
                col - self.height + 1: col + 1, :]
            observation = np.rot90(observation, k=3)
        else:
            raise ValueError()

        assert observation.shape == self.observation_shape

        return observation

    def _step_wrt_orientation(self, action, orientation):
        step_action_list = deque(copy.deepcopy(self.config.step_action_list))
        step_action_list.rotate(-orientation)
        return step_action_list[action]
