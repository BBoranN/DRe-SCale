import os
from env import Environment as env
from sb3_contrib import RecurrentPPO


logdir = 'logs'
en = env()
models_dir = "models/PPO_LSTM_Run3"

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# PPO agent with LSTM
model = RecurrentPPO("MlpLstmPolicy", 
                     env=en, verbose=1, 
                     n_steps=32, n_epochs=10,
                     stats_window_size=100, 
                     batch_size=32,
                     policy_kwargs={'n_lstm_layers': 1, 
                                    'lstm_hidden_size': 256,
                                    'net_arch': dict(pi=[64, 64], vf=[64, 64])},
                     tensorboard_log=logdir
                     )

TIMESTEPS = 80
print("Starting training...")

for i in range(1, 7):
    model.learn(total_timesteps=TIMESTEPS, 
            reset_num_timesteps=False,
            tb_log_name="PPO_LSTM_Run3")
    
    # Define the path properly
    save_path = f'{models_dir}/{TIMESTEPS*i}'
    
    model.save(save_path)
    print(f"Model saved at {save_path}.zip (Time elapsed: ~{i*10} mins)")
