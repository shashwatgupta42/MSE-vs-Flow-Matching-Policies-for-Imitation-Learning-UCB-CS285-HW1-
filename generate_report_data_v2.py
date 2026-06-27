"""Generate all report data: per-seed videos and inference-step sweep.

Outputs:
  exp/report_videos/mse_seed{0..4}.mp4       – individual MSE rollouts (best checkpoint)
  exp/report_videos/flow_seed{0..4}.mp4       – individual Flow rollouts (best checkpoint, 50 steps)
  exp/inference_steps_sweep.json              – {steps: {mean, std, rewards[]}} over 100 episodes
"""

import json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
from PIL import Image

from hw1_imitation.data import download_pusht, load_pusht_zarr, Normalizer
from hw1_imitation.model import MSEPolicy, FlowMatchingPolicy

ENV_ID = "gym_pusht/PushT-v0"


def resize_frame(frame, size):
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized)


def save_single_episode_video(model, normalizer, device, num_steps, output_path, seed=0):
    """Save a single-episode rollout video for one seed."""
    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    chunk_size = model.chunk_size

    obs, _ = env.reset(seed=seed)
    done = False
    chunk_index = chunk_size
    action_chunk = None
    frames = []

    while not done:
        if action_chunk is None or chunk_index >= chunk_size:
            state_t = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
            with torch.no_grad():
                pred_chunk = model.sample_actions(
                    state_t.unsqueeze(0), num_steps=num_steps
                ).cpu().numpy()[0]
            action_chunk = normalizer.denormalize_action(pred_chunk)
            action_chunk = np.clip(action_chunk, action_low, action_high)
            chunk_index = 0

        action = action_chunk[chunk_index]
        obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        frame = env.render()
        frame = resize_frame(frame, (256, 256))
        frames.append(frame)
        done = terminated or truncated
        chunk_index += 1

    env.close()

    with imageio.get_writer(str(output_path), fps=20, codec="libx264", macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(f)
    print(f"  Saved {output_path} ({len(frames)} frames)")


def run_step_sweep(model, normalizer, device, num_episodes=100):
    """Run inference step sweep with 100 episodes for statistical stability."""
    steps_to_test = [1, 3, 5, 10, 20, 50, 100]
    results = {}

    env = gym.make(ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    chunk_size = model.chunk_size

    for steps in steps_to_test:
        print(f"  Evaluating {steps} inference steps over {num_episodes} episodes...")
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
                        pred_chunk = model.sample_actions(
                            state_t.unsqueeze(0), num_steps=steps
                        ).cpu().numpy()[0]
                    action_chunk = normalizer.denormalize_action(pred_chunk)
                    action_chunk = np.clip(action_chunk, action_low, action_high)
                    chunk_index = 0

                action = action_chunk[chunk_index]
                obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                max_reward = max(max_reward, float(reward))
                done = terminated or truncated
                chunk_index += 1
            rewards.append(max_reward)

        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        results[str(steps)] = {"mean": mean_r, "std": std_r, "rewards": rewards}
        print(f"    Steps={steps:>3d} | Mean={mean_r:.4f} ± {std_r:.4f}")

    env.close()
    return results


def main():
    output_dir = Path("exp/report_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    mse_ckpt = Path("exp/seed_42_20260622_195305/wandb/files/checkpoints/checkpoint_step_70000.pkl")
    flow_ckpt = Path("exp/seed_42_20260626_141344/wandb/files/checkpoints/checkpoint_step_70000.pkl")

    # Load normalizer
    print("[1/4] Loading normalizer...")
    zarr_path = download_pusht(Path("data"))
    states, actions, _ = load_pusht_zarr(zarr_path)
    normalizer = Normalizer.from_data(states, actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Generate per-seed MSE videos
    print("[2/4] Generating per-seed MSE videos (best checkpoint step 70000)...")
    mse_model = torch.load(mse_ckpt, map_location=device, weights_only=False)
    for seed in range(5):
        save_single_episode_video(
            mse_model, normalizer, device, num_steps=10,
            output_path=output_dir / f"mse_seed{seed}.mp4", seed=seed,
        )

    # Generate per-seed Flow videos
    print("[3/4] Generating per-seed Flow videos (best checkpoint step 70000, 50 inference steps)...")
    flow_model = torch.load(flow_ckpt, map_location=device, weights_only=False)
    for seed in range(5):
        save_single_episode_video(
            flow_model, normalizer, device, num_steps=50,
            output_path=output_dir / f"flow_seed{seed}.mp4", seed=seed,
        )

    # Run inference steps sweep
    print("[4/4] Running inference steps sweep (100 episodes)...")
    sweep_results = run_step_sweep(flow_model, normalizer, device, num_episodes=100)
    with open("exp/inference_steps_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    print("  Sweep results saved to exp/inference_steps_sweep.json")

    # Print summary
    print("\n=== Summary ===")
    for steps, data in sweep_results.items():
        print(f"  {steps:>3s} steps: {data['mean']:.4f} ± {data['std']:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
