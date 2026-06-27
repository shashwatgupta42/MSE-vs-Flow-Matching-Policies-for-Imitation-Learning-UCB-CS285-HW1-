import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gym_pusht

from hw1_imitation.data import download_pusht, load_pusht_zarr, Normalizer

ENV_ID = "gym_pusht/PushT-v0"
NUM_EVAL_EPISODES = 100

def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of flow inference steps.")
    args = parser.parse_args()

    # Load normalizer data
    print("Loading normalizer from Push-T dataset...")
    # Default data dir matches train config
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("src/hw1_imitation/data")
    zarr_path = download_pusht(data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint_path} onto {device}...")
    model = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.eval()

    print(f"Starting evaluation of {NUM_EVAL_EPISODES} episodes with flow_num_steps={args.num_steps}...")
    
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    rewards = []
    chunk_size = model.chunk_size

    for ep_idx in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk = None
        max_reward = 0.0

        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state_t = (
                    torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                )
                with torch.no_grad():
                    pred_chunk = (
                        model.sample_actions(
                            state_t.unsqueeze(0), num_steps=args.num_steps
                        )
                        .cpu()
                        .numpy()[0]
                    )
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(
                action.astype(np.float32)
            )
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1

        rewards.append(max_reward)
        if (ep_idx + 1) % 10 == 0:
            print(f"Episode {ep_idx + 1}/{NUM_EVAL_EPISODES} | Current mean reward: {np.mean(rewards):.4f}")

    env.close()
    final_mean_reward = np.mean(rewards)
    print("=" * 50)
    print(f"Final Mean Reward (Success Rate) with {args.num_steps} steps: {final_mean_reward:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
