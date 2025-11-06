import embodied

class Arm(embodied.Env):
    def __init__(self, task):
        

    @property
    def obs_space(self):
        return {}

    @property
    def act_space(self):
        return {}

    def step(self, action):
        return 0

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        return dict()
