from __future__ import annotations

"""
HyperPhoenixCV - Resumable hyperparameter search with checkpoint support.

This module provides the HyperPhoenixCV class, which extends the functionality
of scikit-learn's GridSearchCV by adding checkpoint support, random search,
and Bayesian optimization to accelerate the search for optimal hyperparameters.
"""

import os
import random
import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

class HyperPhoenixCV(BaseEstimator):
    """
    Resumable hyperparameter search with checkpoint support and Bayesian optimization.
    Supports exhaustive grid search, random search, and Bayesian optimization.

    Example usage:
    # Create an instance
    hp = HyperPhoenixCV(
        estimator=combat_pipeline,
        param_grid={
            'tfidf__max_features': [8000, 12000, 15000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.001, 0.01, 0.1],
            'clf__penalty': ['l1','l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        },
        scoring=['f1', 'accuracy'],
        cv=5,
        n_jobs=-2,
        checkpoint_path="experiment_checkpoint.pkl",
        results_csv="experiment_results.csv",
        verbose=True
    )

    # Start the search
    hp.fit(X, y)

    # If the process was interrupted, run again with the same checkpoint_path:
    hp.fit(X, y)  # Will continue from the last saved point!

    # Get results
    print("Best parameters:", hp.best_params_)
    print("Best score:", hp.best_score_)

    # Top-10 results
    top_10 = hp.get_top_results(10)
    print(top_10)

    # Manually delete checkpoint
    hp.clear_checkpoint()
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        scoring: str | list[str] = 'f1',
        cv: int = 5,
        n_jobs: int = 1,
        checkpoint_path: str = "hyperphoenix_checkpoint.pkl",
        results_csv: str = "hyperphoenix_results.csv",
        verbose: bool = True,
        clear_checkpoint: bool = False,
        random_search: bool = False,
        n_iter: int = 10,
        random_state: int | None = None,
        use_bayesian_optimization: bool = False,
        bayesian_optimizer = None,
        refit: bool = True,
    ):
        """
        Initializes HyperPhoenixCV.

        Parameters:
        -----------
        estimator : sklearn estimator
            Model/pipeline for hyperparameter tuning
        param_grid : dict
            Dictionary of parameters to search over
        scoring : str or list of str
            Metrics for evaluation (e.g., 'f1', 'accuracy' or ['f1', 'accuracy'])
        cv : int
            Number of folds for cross-validation
        n_jobs : int
            Number of processes for parallel computation
        checkpoint_path : str
            Path to checkpoint file
        results_csv : str
            Path to CSV file for results
        verbose : bool
            Whether to print progress
        clear_checkpoint : bool
            Whether to delete existing checkpoint on initialization
        random_search : bool
            Whether to use random search instead of exhaustive grid search
        n_iter : int
            Number of random combinations (if random_search=True)
        random_state : int, optional
            Random seed for reproducibility
        use_bayesian_optimization : bool
            Whether to use Bayesian optimization (predictive parameter selection)
        bayesian_optimizer : sklearn regressor, optional
            Model that predicts which parameters will perform better
            (defaults to RandomForestRegressor)
        refit : bool, default=True
            Whether to refit the best model on the entire dataset after search.
            If True, after hyperparameter search completes, `best_estimator_.fit(X, y)` will be called.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
        self.cv = cv
        self.n_jobs = n_jobs
        self.checkpoint_path = checkpoint_path
        self.results_csv = results_csv
        self.verbose = verbose
        self.random_search = random_search
        self.n_iter = n_iter
        self.random_state = random_state
        self.use_bayesian_optimization = use_bayesian_optimization
        self.bayesian_optimizer = (
            bayesian_optimizer or RandomForestRegressor(n_estimators=20, random_state=42)
        )
        self.label_encoders = {}
        self.refit = refit

        # Delete checkpoint if specified
        if clear_checkpoint and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if self.verbose:
                print(f"Deleted checkpoint: {checkpoint_path}")

    def _generate_param_list(self) -> list[dict]:
        """
        Generates a list of parameters: exhaustive grid search or random.

        Returns:
        --------
        list[dict]: List of parameter combinations to test.
        """
        all_params = list(ParameterGrid(self.param_grid))

        if self.random_search:
            if self.random_state is not None:
                random.seed(self.random_state)

            if len(all_params) <= self.n_iter:
                if self.verbose:
                    print(
                        f"Всего комбинаций ({len(all_params)}) <= n_iter ({self.n_iter}). "
                        f"Используем все."
                    )
                return all_params
            # else is unnecessary after return
            selected_params = random.sample(all_params, self.n_iter)
            if self.verbose:
                print(
                    f"Выбрано {len(selected_params)} случайных комбинаций "
                    f"из {len(all_params)} возможных."
                )
            return selected_params
        else:
            if self.verbose:
                print(f"Полный перебор: {len(all_params)} комбинаций.")
            return all_params

    def _load_checkpoint(self) -> list[dict]:
        """
        Loads results from a checkpoint.

        Returns:
        --------
        list[dict]: List of results from the checkpoint.
        """
        if os.path.exists(self.checkpoint_path):
            results = joblib.load(self.checkpoint_path)
            if self.verbose:
                print(f"Loaded {len(results)} completed combinations from checkpoint.")
                # Display current best results from checkpoint
                if results:
                    valid_results = [r for r in results if 'error' not in r]
                    if valid_results:
                        # Sort by the first metric
                        best_result = max(valid_results,
                                          key=lambda x: x.get(f'mean_test_{self.scoring[0]}',
                                                              float('-inf')))
                        print(f"Current best result from checkpoint:")
                        score_key = f'mean_test_{self.scoring[0]}'
                        std_key = f'std_test_{self.scoring[0]}'
                        print(f"   score: {best_result.get(score_key, 0):.4f} ± "
                              f"{best_result.get(std_key, 0):.4f}")
                        if len(self.scoring) > 1:
                            for metric in self.scoring[1:]:
                                mean_key = f'mean_test_{metric}'
                                std_key = f'std_test_{metric}'
                                if mean_key in best_result and std_key in best_result:
                                    print(f"   {metric}: {best_result[mean_key]:.4f} ± "
                                          f"{best_result[std_key]:.4f}")
                        print(f"   Parameters: {best_result.get('params', {})}")
            return results
        return []

    def _save_checkpoint(self, results: list[dict]):
        """
        Saves results to a checkpoint.

        Parameters:
        -----------
        results : list[dict]
            List of results to save.
        """
        joblib.dump(results, self.checkpoint_path)

    def _format_scores(self, cv_results: dict[str, np.ndarray]) -> dict[str, any]:
        """
        Formats cross-validation results.

        Parameters:
        -----------
        cv_results : dict[str, np.ndarray]
            Cross-validation results from cross_validate.

        Returns:
        --------
        dict[str, any]: Formatted results.
        """
        scores = {}
        for metric in self.scoring:
            test_metric = f'test_{metric}'
            if test_metric in cv_results:
                scores[f'mean_test_{metric}'] = float(cv_results[test_metric].mean())
                scores[f'std_test_{metric}'] = float(cv_results[test_metric].std())
                scores[f'scores_{metric}'] = cv_results[test_metric].tolist()
        return scores

    def _encode_params(self, params_list: list[dict]) -> np.ndarray:
        """
        Encodes a list of parameters into a numeric matrix.

        Parameters:
        -----------
        params_list : list[dict]
            List of parameters to encode.

        Returns:
        --------
        np.ndarray: Encoded parameters as a matrix.
        """
        if not params_list:
            return np.array([]).reshape(0, -1)

        df = pd.DataFrame(params_list)
        X = df.copy()

        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        return X.values

    def _suggest_next_params(
        self,
        all_param_combinations: list[dict],
        completed_results: list[dict],
    ) -> list[dict]:
        """
        Sorts remaining parameters by predicted metric (if Bayesian optimization is used).

        Parameters:
        -----------
        all_param_combinations : list[dict]
            All possible parameter combinations.
        completed_results : list[dict]
            Already completed results.

        Returns:
        --------
        list[dict]: Sorted list of parameters.
        """
        if not self.use_bayesian_optimization or not completed_results:
            return all_param_combinations

        # Train the model on completed results
        completed_params = [r['params'] for r in completed_results]
        completed_scores = [r[f'mean_test_{self.scoring[0]}'] for r in completed_results]

        X_train = self._encode_params(completed_params)
        y_train = np.array(completed_scores)

        if X_train.size == 0 or len(y_train) == 0:
            return all_param_combinations

        self.bayesian_optimizer.fit(X_train, y_train)

        # Predict for remaining
        X_remaining = self._encode_params(all_param_combinations)
        if X_remaining.size == 0:
            return all_param_combinations

        predicted_scores = self.bayesian_optimizer.predict(X_remaining)

        # Sort by descending predicted score
        sorted_indices = np.argsort(predicted_scores)[::-1]
        return [all_param_combinations[i] for i in sorted_indices]

    def fit(self, X, y, groups=None):
        """
        Performs hyperparameter tuning with intermediate result saving.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Groups for group cross-validation (if used).

        Returns:
        --------
        self : object
            Returns the instance.
        """
        # Load progress
        all_results = self._load_checkpoint()

        # Generate all parameter combinations (exhaustive or random)
        param_list = self._generate_param_list()
        if self.verbose:
            print(f"Total combinations: {len(param_list)}")

        # Exclude already processed
        completed_params = [r['params'] for r in all_results if 'params' in r]
        remaining_params = [p for p in param_list if p not in completed_params]
        if self.verbose:
            print(f"Remaining to process: {len(remaining_params)}")

        # If Bayesian optimization is used, sort remaining parameters by prediction
        if self.use_bayesian_optimization:
            remaining_params = self._suggest_next_params(remaining_params, all_results)
            if self.verbose:
                print("Remaining parameters sorted by predicted metric.")

        # --- Determine CV ---

        if isinstance(self.cv, int):
            classification_metrics = {
                'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall',
                'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro',
                'precision_micro', 'precision_weighted', 'recall_macro',
                'recall_micro', 'recall_weighted', 'jaccard', 'roc_auc'
            }

            if any(m in classification_metrics for m in self.scoring):
                cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = self.cv
        # --- CV determined ---

        # Iterate over remaining parameters
        for i, params in enumerate(remaining_params, start=1):
            if self.verbose:
                print(f"\n[{i}/{len(remaining_params)}] Testing: {params}")

            try:
                estimator_with_params = self.estimator.set_params(**params)

                scores = cross_validate(
                    estimator_with_params, X, y,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )

                result = {
                    'params': params,
                    **self._format_scores(scores)
                }
                all_results.append(result)

                # Update Bayesian optimization model if used
                if self.use_bayesian_optimization:
                    # Not necessary to update each time — can be done every N iterations
                    pass  # Model is updated on the next call to _suggest_next_params

                self._save_checkpoint(all_results)

                if self.verbose:
                    current_metrics = []
                    for metric in self.scoring:
                        mean_key = f'mean_test_{metric}'
                        std_key = f'std_test_{metric}'
                        if mean_key in result and std_key in result:
                            current_metrics.append(
                                f"{metric}: {result[mean_key]:.4f} ± {result[std_key]:.4f}"
                            )
                    current_str = " | ".join(current_metrics)

                    metric_key = f'mean_test_{self.scoring[0]}'
                    if metric_key in result:
                        best_score = max(
                            r[metric_key] for r in all_results if metric_key in r
                        )
                        best_metrics = []
                        for metric in self.scoring:
                            metric_key_other = f'mean_test_{metric}'
                            if metric_key_other in result:
                                best_other = max(
                                    r[metric_key_other]
                                    for r in all_results
                                    if metric_key_other in r
                                )
                                best_metrics.append(f"{metric}: {best_other:.4f}")
                        best_str = " | ".join(best_metrics)
                        print(f"Saved. Current: {current_str} | Best: {best_str}")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error: {e}")
                all_results.append({
                    'params': params,
                    'error': str(e)
                })
                self._save_checkpoint(all_results)

        # Save results to CSV
        self._save_results_to_csv(all_results)

        # Save attributes for compatibility with GridSearchCV
        self.cv_results_ = self._format_cv_results(all_results)
        self.best_params_ = self._get_best_params(all_results)
        self.best_score_ = self._get_best_score(all_results)

        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y)

        if self.verbose:
            print(f"\nAll results saved to {self.results_csv}")
            print(f"Best result ({self.scoring[0]}): {self.best_score_:.4f}")
            if self.random_search:
                total_grid = len(list(ParameterGrid(self.param_grid)))
                print(
                    f"Random search used: {self.n_iter} out of {total_grid} "
                    f"possible combinations ({self.n_iter/total_grid*100:.2f}%)"
                )

        return self

    def predict(self, X):
        """
        Predictions using the best model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for prediction.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Class probabilities (if the best model supports predict_proba).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for prediction.

        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        """
        Evaluate the best model on data X, y.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for evaluation.
        y : array-like of shape (n_samples,)
            True values.

        Returns:
        --------
        score : float
            Metric value (default uses scoring[0]).
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.score(X, y)

    def _format_cv_results(self, results: list[dict]) -> dict[str, np.ndarray]:
        """
        Formats results into a GridSearchCV-compatible format.

        Parameters:
        -----------
        results : list[dict]
            List of results.

        Returns:
        --------
        dict[str, np.ndarray]: Formatted results.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {}

        # Create a dictionary with results
        cv_results = {'params': [r['params'] for r in valid_results]}

        for metric in self.scoring:
            mean_key = f'mean_test_{metric}'
            std_key = f'std_test_{metric}'

            cv_results[mean_key] = np.array([r[mean_key] for r in valid_results])
            cv_results[std_key] = np.array([r[std_key] for r in valid_results])

        return cv_results

    def _get_best_params(self, results: list[dict]) -> dict:
        """
        Retrieves the best parameters.

        Parameters:
        -----------
        results : list[dict]
            List of results.

        Returns:
        --------
        dict: Best parameters.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {}

        # Sort by the first metric
        best_result = max(valid_results,
                         key=lambda x: x[f'mean_test_{self.scoring[0]}'])
        return best_result['params']

    def _get_best_score(self, results: list[dict]) -> float:
        """
        Retrieves the best score.

        Parameters:
        -----------
        results : list[dict]
            List of results.

        Returns:
        --------
        float: Best score.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return 0.0

        best_result = max(valid_results,
                         key=lambda x: x[f'mean_test_{self.scoring[0]}'])
        return best_result[f'mean_test_{self.scoring[0]}']

    def _save_results_to_csv(self, results: list[dict]):
        """
        Saves results to CSV.

        Parameters:
        -----------
        results : list[dict]
            List of results to save.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            df = pd.DataFrame(columns=['params'])
            df.to_csv(self.results_csv, index=False)
            return

        # Build DataFrame
        rows = []
        for r in valid_results:
            row = {}
            # Add parameters as separate columns
            row.update(r['params'])
            # Add metrics
            for metric in self.scoring:
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in r:
                    row[mean_key] = r[mean_key]
                if std_key in r:
                    row[std_key] = r[std_key]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.results_csv, index=False)

    def get_top_results(self, n: int = 10) -> pd.DataFrame:
        """
        Returns top‑N results.

        Parameters:
        -----------
        n : int
            Number of top results to return.

        Returns:
        --------
        pd.DataFrame: Top‑N results.
        """
        if not hasattr(self, 'cv_results_') or not self.cv_results_:
            return pd.DataFrame()

        # Create DataFrame from results
        results = []
        for i in range(len(self.cv_results_['params'])):
            row = {}
            row.update(self.cv_results_['params'][i])
            for metric in self.scoring:
                row[f'mean_test_{metric}'] = self.cv_results_[f'mean_test_{metric}'][i]
                row[f'std_test_{metric}'] = self.cv_results_[f'std_test_{metric}'][i]
            results.append(row)

        df = pd.DataFrame(results)
        # Sort by the first metric
        df = df.sort_values(f'mean_test_{self.scoring[0]}', ascending=False)
        return df.head(n)

    def clear_checkpoint(self):
        """
        Deletes the checkpoint file.
        """
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            if self.verbose:
                print(f"Deleted checkpoint: {self.checkpoint_path}")

    def load_results_from_checkpoint(self, n: int = 10) -> pd.DataFrame:
        """
        Loads results from a checkpoint and returns top‑N.
        Useful when fit() was interrupted and CSV was not created.

        Parameters:
        -----------
        n : int
            Number of top results to return

        Returns:
        --------
        pd.DataFrame
            Top‑N results from the checkpoint
        """
        if not os.path.exists(self.checkpoint_path):
            if self.verbose:
                print(f"⚠️ Checkpoint {self.checkpoint_path} not found.")
            return pd.DataFrame()

        # Load results from checkpoint
        all_results = self._load_checkpoint()
        valid_results = [r for r in all_results if 'error' not in r]

        if not valid_results:
            if self.verbose:
                print("⚠️ No valid results in checkpoint.")
            return pd.DataFrame()

        # Build DataFrame
        rows = []
        for r in valid_results:
            row = {}
            # Add parameters as separate columns
            row.update(r['params'])
            # Add metrics
            for metric in self.scoring:
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in r:
                    row[mean_key] = r[mean_key]
                if std_key in r:
                    row[std_key] = r[std_key]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by the first metric (the one we optimize for)
        df = df.sort_values(f'mean_test_{self.scoring[0]}', ascending=False)

        if self.verbose:
            print(f"Loaded {len(df)} valid results from checkpoint.")
            print(f"Best {self.scoring[0]}: {df.iloc[0][f'mean_test_{self.scoring[0]}']:.4f}")

        return df.head(n)
