import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot training logs from log.csv")
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the log.csv file. If not provided, the most recent log in exp/ will be used."
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if csv_path is None:
        exp_dir = Path("exp")
        if not exp_dir.exists():
            print("No exp/ directory found. Please run training first.")
            return
        
        # Find the most recent directory in exp/
        subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("No experiment directories found in exp/.")
            return
        
        # Sort by modification time
        subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        most_recent_dir = subdirs[0]
        csv_path = most_recent_dir / "log.csv"
        if not csv_path.exists():
            print(f"No log.csv found in the most recent directory: {most_recent_dir}")
            return
        print(f"Using most recent log directory: {most_recent_dir}")
    else:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"File not found: {csv_path}")
            return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter rows
    train_data = df[df["train/loss"].notna()]
    eval_data = df[df["eval/mean_reward"].notna()]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training loss
    if not train_data.empty:
        axes[0].plot(train_data["step"], train_data["train/loss"], label="Train Loss", color="tab:blue")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No Train Loss Data", ha="center", va="center")

    # Plot evaluation reward
    if not eval_data.empty:
        axes[1].plot(eval_data["step"], eval_data["eval/mean_reward"], label="Eval Mean Reward", color="tab:orange", marker="o")
        axes[1].set_title("Evaluation Mean Reward")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean Reward")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No Eval Reward Data", ha="center", va="center")

    plt.tight_layout()
    
    # Save plot
    output_path = csv_path.parent / "metrics_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
