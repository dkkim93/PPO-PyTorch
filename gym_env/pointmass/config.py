import numpy as np


class Config(object):
    def __init__(self):
        super(Config, self).__init__()

        self._set_action_dict()
        self._set_orientation_dict()
        self._set_grid_dict()
        self._set_color_dict()

    def _set_action_dict(self):
        self.step_action_list = [
            np.array([-1, 0]),  # Forward w.r.t orientation up
            np.array([0, 1]),  # Right w.r.t orientation up
            np.array([1, 0]),  # Backward w.r.t orientation up
            np.array([0, -1])  # Left w.r.t orientation up
        ]

        self.action_dict = {
            "stay": np.array([0, 0]),
            "step_forward": 0,
            "step_right": 1,
            "step_backward": 2,
            "step_left": 3,
            "spin_right": +1,
            "spin_left": -1,
        }

    def _set_orientation_dict(self):
        self.orientation_dict = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
        }

        self.orientation_delta_dict = {
            "up": np.array([-1, 0]),
            "right": np.array([0, 1]),
            "down": np.array([1, 0]),
            "left": np.array([+0, -1]),
        }

    def _set_grid_dict(self):
        self.grid_dict = {
            "empty": 0,
            "wall": 1,
            "prey": 2,
            "predator": 3,
            "orientation": 4,
        }

    def _set_color_dict(self):
        self.color_dict = {
            "empty": [0., 0., 0.],  # Black
            "wall": [0.5, 0.5, 0.5],  # Gray
            "own": [0., 0., 1.],  # Blue
            "teammate": [0.6, 0.8, 0.8],  # Light Blue
            "opponent": [1., 0., 0.],  # Red
            "orientation": [0.1, 0.1, 0.1],  # Dark Gray
        }
