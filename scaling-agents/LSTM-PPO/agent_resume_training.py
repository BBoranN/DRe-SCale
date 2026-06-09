import os
import setting
from env import Environment as env
from sb3_contrib import RecurrentPPO

logdir = setting.ppo_lstm_agent_log_dir
models_dir = setting.ppo_lstm_agent_models_dir
en = env()

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# 1. Define the path to your last saved checkpoint
last_checkpoint_path = f'{models_dir}/1600.zip'

print(f"Loading model from {last_checkpoint_path}...")

# 2. Load the model instead of initializing a new one
model = RecurrentPPO.load(
    path=last_checkpoint_path,
    env=en,
    tensorboard_log=logdir
)

TIMESTEPS = 200

# 3. Calculate where you left off
start_iteration = (1600 // TIMESTEPS) + 1

print("Resuming training...")

for i in range(start_iteration, 1000):
    # reset_num_timesteps=False ensures TensorBoard appends to the existing graph
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False,
        tb_log_name="PPO_LSTM_Run_last_final"
    )
    
    # Save the next steps properly (1800, 2000, 2200, etc.)
    save_path = f'{models_dir}/{TIMESTEPS*i}'
    
    model.save(save_path)
    print(f"Model saved at {save_path}.zip (Resumed iteration {i})")