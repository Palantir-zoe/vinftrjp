import os

import dill as pickle

from src.vi_nflows import prepare_flow_for_inference


def train_with_checkpoint(path_dir, folder, mk, func, *args, **kwargs):
    """Train normalizing flow with checkpointing functionality."""
    if len(path_dir) == 0:
        return func(*args, **kwargs)

    _path_dir = os.path.join(path_dir, folder)
    name = f"flow_{mk}".replace(" ", "_")
    file = os.path.join(_path_dir, f"{name}.pickle")

    # Check if pre-trained model exists
    if os.path.exists(file):
        try:
            # Load pre-trained model from disk
            with open(file, "rb") as f:
                flow = pickle.load(f)
            flow = prepare_flow_for_inference(flow, device="cpu")
            print(f"\nLoaded pre-trained model: {file}")
            return flow

        except Exception as e:
            print(f"\nModel loading failed: {e}, initializing training...")

    # Train new model if no pre-trained model exists
    print("\nInitializing training...")
    flow = func(*args, **kwargs)
    flow = prepare_flow_for_inference(flow, device="cpu")

    # Save trained model to disk
    try:
        os.makedirs(_path_dir, exist_ok=True)
        with open(file, "wb") as f:
            pickle.dump(flow, f)
        print(f"Model saved successfully: {file}\n")
    except Exception as e:
        print(f"Model saving failed: {e}\n")

    return flow
