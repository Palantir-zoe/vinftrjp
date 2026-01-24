import argparse
import os
import subprocess
import sys


def setup_argparse():
    parser = argparse.ArgumentParser(description="Process start and end parameters")
    parser.add_argument("--start", type=int, nargs="?", default=1, help="Start value (default: 1)")
    parser.add_argument("--end", type=int, nargs="?", default=3, help="End value (default: 3)")
    return parser.parse_args()


def main():
    # Exit immediately if any command fails
    try:
        # Get project root directory and change to it
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print("Error: Not in a git repository or git not found")
            sys.exit(1)

        project_dir = result.stdout.strip()
        os.chdir(project_dir)

        # List of tasks to execute
        tasks = [
            "scripts/vs/run_smc.py",
            "scripts/vs/run_vs_rjbridge_vinfs.py",
            "scripts/vs/run_vs_rjbridge_vicnfs.py",
            "scripts/vs/run_vs_rjbridge_vinfs_ablation.py",
            "scripts/vs/run_vs_rjbridge_vicnfs_abalation.py",
        ]

        # Get command line arguments
        args = setup_argparse()

        # Execute main tasks
        for task in tasks:
            print(f"Executing: {task}")

            cmd = ["uv", "run", task, f"--start={args.start}", f"--end={args.end}"]
            result = subprocess.run(cmd, check=False)

            if result.returncode != 0:
                print(f"Failed to execute: {task}")
                sys.exit(1)

        print("All tasks completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
