import os
import setting
from env import Environment as env
from sb3_contrib import RecurrentPPO


logdir = setting.ppo_lstm_agent_log_dir
en = env()
models_dir = setting.ppo_lstm_agent_models_dir

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# PPO agent with LSTM
model = RecurrentPPO("MlpLstmPolicy", 
                     env=en, verbose=1, 
                     n_steps=128, n_epochs=10,
                     stats_window_size=100, 
                     batch_size=128,
                     policy_kwargs={'n_lstm_layers': 1, 
                                    'lstm_hidden_size': 256,
                                    'net_arch': dict(pi=[64, 64], vf=[64, 64])},
                     tensorboard_log=logdir
                     )

TIMESTEPS = 1000
print("Starting training...")

for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, 
            reset_num_timesteps=False,
            tb_log_name="PPO_LSTM_Run3_cpu")
    
    # Define the path properly
    save_path = f'{models_dir}/{TIMESTEPS*i}'
    
    model.save(save_path)
    print(f"Model saved at {save_path}.zip (Time elapsed: ~{i*10} mins)")
