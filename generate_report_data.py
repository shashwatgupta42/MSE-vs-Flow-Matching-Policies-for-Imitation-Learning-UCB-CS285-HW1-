import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gym_pusht
import imageio.v2 as imageio
from PIL import Image

from hw1_imitation.data import download_pusht, load_pusht_zarr, Normalizer
from hw1_imitation.model import MSEPolicy, FlowMatchingPolicy

ENV_ID = "gym_pusht/PushT-v0"

def resize_frame(frame, size):
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized)

def save_rollout_video(model, normalizer, device, num_steps, output_path, num_episodes=5):
    print(f"Generating rollout video for {output_path}...")
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high

    all_frames = []
    chunk_size = model.chunk_size

    for ep_idx in range(num_episodes):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk = None

        # Add a text watermark or indicator for start of episode if possible, or just play them
        # Let's collect frames
        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                state_t = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                with torch.no_grad():
                    if isinstance(model, FlowMatchingPolicy):
                        pred_chunk = model.sample_actions(state_t.unsqueeze(0), num_steps=num_steps).cpu().numpy()[0]
                    else:
                        pred_chunk = model.sample_actions(state_t.unsqueeze(0)).cpu().numpy()[0]
                action_chunk = normalizer.denormalize_action(pred_chunk)
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0

            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            
            frame = env.render()
            frame = resize_frame(frame, (256, 256))
            all_frames.append(frame)
            done = terminated or truncated
            chunk_index += 1

    env.close()

    # Save frames as mp4
    with imageio.get_writer(output_path, fps=20, codec="libx264", macro_block_size=1) as writer:
        for f in all_frames:
            writer.append_data(f)
    print(f"Video saved to {output_path}")

def run_step_sweep(model, normalizer, device):
    steps_to_test = [5, 10, 20, 50, 100]
    results = {}
    
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    chunk_size = model.chunk_size
    num_episodes = 50  # 50 episodes is fast enough but statistically representative

    for steps in steps_to_test:
        print(f"Evaluating {steps} inference steps over {num_episodes} episodes...")
        rewards = []
        for ep_idx in range(num_episodes):
            obs, _ = env.reset(seed=ep_idx)
            done = False
            chunk_index = chunk_size
            action_chunk = None
            max_reward = 0.0

            while not done:
                if action_chunk is None or chunk_index >= chunk_size:
                    state_t = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
                    with torch.no_grad():
                        pred_chunk = model.sample_actions(state_t.unsqueeze(0), num_steps=steps).cpu().numpy()[0]
                    action_chunk = normalizer.denormalize_action(pred_chunk)
                    action_chunk = np.clip(action_chunk, action_low, action_high)
                    chunk_index = 0

                action = action_chunk[chunk_index]
                obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                max_reward = max(max_reward, float(reward))
                done = terminated or truncated
                chunk_index += 1
            rewards.append(max_reward)
        
        results[str(steps)] = float(np.mean(rewards))
        print(f"Steps: {steps} | Mean Reward: {results[str(steps)]:.4f}")

    env.close()
    return results

def main():
    # Setup directories
    output_dir = Path("exp")
    output_dir.mkdir(exist_ok=True)

    # Paths to checkpoints
    mse_checkpoint_path = Path("exp/seed_42_20260622_195305/wandb/files/checkpoints/checkpoint_step_70000.pkl")
    flow_checkpoint_path = Path("exp/seed_42_20260626_141344/wandb/files/checkpoints/checkpoint_step_70000.pkl")

    # Load normalizer
    print("Loading normalizer...")
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("src/hw1_imitation/data")
    zarr_path = download_pusht(data_dir)
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate MSE video
    if mse_checkpoint_path.exists():
        print("Loading MSE model...")
        mse_model = torch.load(mse_checkpoint_path, map_location=device, weights_only=False)
        save_rollout_video(mse_model, normalizer, device, num_steps=0, output_path=output_dir / "best_mse_rollouts.mp4")
    else:
        print(f"MSE checkpoint not found at {mse_checkpoint_path}")

    # 2. Generate Flow video and Sweep steps
    if flow_checkpoint_path.exists():
        print("Loading Flow model...")
        flow_model = torch.load(flow_checkpoint_path, map_location=device, weights_only=False)
        
        # Save video for flow model with 50 steps
        save_rollout_video(flow_model, normalizer, device, num_steps=50, output_path=output_dir / "best_flow_rollouts.mp4")
        
        # Run sweep
        sweep_results = run_step_sweep(flow_model, normalizer, device)
        with open(output_dir / "inference_steps_results.json", "w") as f:
            json.dump(sweep_results, f)
        print("Sweep results saved successfully.")
    else:
        print(f"Flow checkpoint not found at {flow_checkpoint_path}")

if __name__ == "__main__":
    main()
