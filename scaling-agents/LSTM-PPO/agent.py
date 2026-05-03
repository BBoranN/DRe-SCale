import argparse
import os
from env import Environment as env
from sb3_contrib import RecurrentPPO


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM-PPO scaling agent training")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="TensorBoard / training log directory (tensorboard_log)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/PPO_LSTM_Run1",
        help="Directory for saved model checkpoints",
    )
    parser.add_argument(
        "--tb-log-name",
        type=str,
        default="PPO_LSTM_Run3_cpu",
        help="Run name passed to model.learn(tb_log_name=...)",
    )
    return parser.parse_args()


args = parse_args()
logdir = args.logdir
models_dir = args.models_dir

# Utility to create directories
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

en = env()

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
            tb_log_name=args.tb_log_name)

    # Define the path properly
    save_path = f'{models_dir}/{TIMESTEPS*i}'

    model.save(save_path)
    print(f"Model saved at {save_path}.zip (Time elapsed: ~{i*10} mins)")
