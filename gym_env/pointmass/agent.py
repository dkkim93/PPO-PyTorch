class Agent(object):
    def __init__(self):
        self.position = None
        self.vel = None
        self.orientation = None
        self.color = None

# class Agent(object):
#     def __init__(self, i_agent, agent_type, base_gridmap_array):
#         self.id = i_agent
#         self.type = agent_type
#         self.base_gridmap_array = base_gridmap_array
#         self.config = Config()
#
#     @property
#     def location(self):
#         return self._location
#
#     @location.setter
#     def location(self, value):
#         grid = self.base_gridmap_array[value[0], value[1]]
#         if grid != self.config.grid_dict["wall"]:
#             self._location = value
#
#     @property
#     def orientation(self):
#         return self._orientation
#
#     @orientation.setter
#     def orientation(self, value):
#         self._orientation = value % len(self.config.orientation_dict)
#
#     @property
#     def orientation_location(self):
#         orientation = list(self.config.orientation_dict.keys())[self._orientation]
#         orientation_delta = self.config.orientation_delta_dict[orientation]
#         self._orientation_location = self.location + orientation_delta
#
#         grid = self.base_gridmap_array[self._orientation_location[0], self._orientation_location[1]]
#         if grid == self.config.grid_dict["wall"]:
#             self._orientation_location = np.copy(self.location)
#
#         return self._orientation_location
