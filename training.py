import gym
import basketball_gym
import mujoco
import mujoco_viewer
from stable_baselines3 import SAC

# Register the custom environment
basketball_gym.basketball_env.register_env()

# Create the environment directly
env = gym.make('BasketballEnv-v0', xml_file_path='test.xml')

# Load the environment's MuJoCo model and data for visualization
xml_file_path = 'test.xml'
mujoco_model = mujoco.MjModel.from_xml_path(xml_file_path)
mujoco_data = mujoco.MjData(mujoco_model)

# Create a MuJoCo viewer for live visualization
viewer = mujoco_viewer.MujocoViewer(mujoco_model, mujoco_data)

# Initialize the SAC model for training
model = SAC("MlpPolicy", env, verbose=1)

# Train the model while watching through the viewer
total_timesteps = 100000
current_timesteps = 0
obs, _ = env.reset()  # Unpack only the observation from the reset method
terminated = truncated = False

while current_timesteps < total_timesteps:
    if not (terminated or truncated):
        # Predict the next action using the current model
        action, _ = model.predict(obs, deterministic=True)

        # Take the action and observe the results
        obs, reward, terminated, truncated, info = env.step(action)

        # Update the MuJoCo data for visualization
        mujoco_data.ctrl[:] = action
        mujoco.mj_step(mujoco_model, mujoco_data)

        # Render the current frame
        viewer.render()

        current_timesteps += 1
    else:
        # Reset the environment if the episode is done or truncated
        obs, _ = env.reset()  # Unpack only the observation from the reset method
        terminated = truncated = False

# Close the viewer after training is complete
viewer.close()

# Save the trained model for future use
model.save("sac_basketball_agent")
