"""
Checkpoint manager for saving and loading intermediate results.
"""

import os
import joblib
from typing import List, Dict, Any


class CheckpointManager:
    """
    Manages checkpoint files for hyperparameter search.
    """

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

    def load(self) -> List[Dict[str, Any]]:
        """
        Load results from checkpoint file.

        Returns:
            List of results (each result is a dict with at least 'params' key).
        """
        if os.path.exists(self.checkpoint_path):
            results = joblib.load(self.checkpoint_path)
            if self.verbose:
                print(f"Loaded {len(results)} completed combinations from checkpoint.")
            return results
        if self.verbose:
            print(f"No checkpoint found at {self.checkpoint_path}.")
        return []

    def save(self, results: List[Dict[str, Any]]):
        """
        Save results to checkpoint file.

        Args:
            results: List of results to save.
        """
        joblib.dump(results, self.checkpoint_path)
        if self.verbose:
            print(f"Checkpoint saved to {self.checkpoint_path}")

    def clear(self):
        """
        Delete the checkpoint file if it exists.
        """
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            if self.verbose:
                print(f"Deleted checkpoint: {self.checkpoint_path}")
        elif self.verbose:
            print(f"Checkpoint {self.checkpoint_path} does not exist.")