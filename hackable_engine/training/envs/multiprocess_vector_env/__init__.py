
class RayEnvFactory:
    def get_ray_env(self):
        self.env = None
        self.env_config = {}
        self.observation_space = None
        self.action_space = None
        self.clip_rewards = None
        self.normalize_actions = True
        self.clip_actions = False
        self._is_atari = None
        self.disable_env_checking = False
        # Deprecated settings:
        self.render_env = False
        self.action_mask_key = "action_mask"

        # `self.env_runners()`
        self.env_runner_cls = None
        self.num_env_runners = 0
        self.num_envs_per_env_runner = 1
