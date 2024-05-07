import gym
import numpy as np
import mujoco
from gym import spaces

class BasketballEnv(gym.Env):
    """Basketball Environment with a human and basketball."""

    metadata = {'render.modes': ['human']}

    def __init__(self, xml_file_path, control_freq=1000, target_fps=30):
        super().__init__()
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)

        # Action space: range from -1 to 1 for each actuator
        num_actuators = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float32)

        # Observation space includes agent states, ball, and hoop data relative to the agent
        num_positions = self.model.nq
        num_velocities = self.model.nv
        ball_pos_dim = 3
        hoop_pos_dim = 3
        total_size = num_positions + num_velocities + ball_pos_dim + hoop_pos_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)

        # Access body objects directly by name using named access API
        self.ball_body = self.model.body("basketball")
        self.hoop_body = self.model.body("basketballhoop")

        # Define the number of steps required to achieve 30 FPS
        self.sim_steps_per_frame = control_freq // target_fps

    def _get_observation(self):
        # Retrieve agent's joints' positions and velocities
        joints = np.concatenate([self.data.qpos, self.data.qvel])

        # Retrieve ball and hoop positions relative to the agent's base
        agent_base_pos = self.data.qpos[:3]  # Assume the base is at the first 3 positions

        ball_body_data = self.data.body(self.ball_body.name)
        hoop_body_data = self.data.body(self.hoop_body.name)

        ball_pos_rel = ball_body_data.xpos - agent_base_pos
        hoop_pos_rel = hoop_body_data.xpos - agent_base_pos

        # Combine all relevant data into a single observation
        return np.concatenate([joints, ball_pos_rel, hoop_pos_rel])

    def _get_info(self):
        # Retrieve positions of the ball and hoop
        ball_pos = self.data.body(self.ball_body.name).xpos
        hoop_pos = self.data.body(self.hoop_body.name).xpos

        # Calculate the distance between the ball and the hoop
        distance = np.linalg.norm(ball_pos - hoop_pos)

        return {'distance_to_hoop': distance}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = np.zeros(self.model.nv)

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        # Obtain the initial observation and info
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Apply the action to the environment
        self.data.ctrl[:] = action
        # Run simulation steps to reach the target frame rate
        for _ in range(self.sim_steps_per_frame):
            mujoco.mj_step(self.model, self.data)

        # Retrieve the new observation and info
        observation = self._get_observation()
        info = self._get_info()

        # Compute reward with additional milestones
        reward = self.compute_reward(info['distance_to_hoop'], observation)

        # Determine whether the episode is terminated or truncated
        terminated = self.check_if_done(info['distance_to_hoop'])
        truncated = False  # You can set logic here to handle the "max_episode_steps" limit

        return observation, reward, terminated, truncated, info

    def compute_reward(self, distance_to_hoop, observation):
        return -distance_to_hoop

    def check_if_done(self, distance_to_hoop):
        # Check if the ball is close enough to the hoop
        return distance_to_hoop < 0.1

# Register the environment
def register_env():
    from gym.envs.registration import register

    register(
        id='BasketballEnv-v0',
        entry_point='basketball_gym.basketball_env:BasketballEnv',
        max_episode_steps=1000,
    )
