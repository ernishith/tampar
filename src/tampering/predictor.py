import os
import sys
from pathlib import Path
from typing import Tuple

root_dir = Path(os.getcwd()).parent.parent.parent.parent
sys.path.append(root_dir.as_posix())

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from src.tampering.evaluate import evaluate


class TamperingClassificator:
    def __init__(self, model_name: str, model_parameters=None):
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.test_split_size = 0.3

    def set_data(self, X, y, ids):
        self.X, self.y = X, y
        self.ids = ids

    def build_model(self):
        """Build classifier model based on model_name."""
        if self.model_name == "simple_threshold":
            # Original: Single threshold (depth=1 decision tree)
            params = {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": 1,
                "random_state": 42,
            }
            if self.model_parameters is not None:
                params.update(self.model_parameters)
            return DecisionTreeClassifier(**params)

        elif self.model_name == "decision_tree":
            # Better decision tree with reasonable depth
            params = {
                "criterion": "gini",
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
            }
            if self.model_parameters is not None:
                params.update(self.model_parameters)
            return DecisionTreeClassifier(**params)

        elif self.model_name == "random_forest":
            # Random Forest - ensemble of decision trees
            params = {
                "n_estimators": 100,
                "criterion": "gini",
                "max_depth": 8,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            }
            if self.model_parameters is not None:
                params.update(self.model_parameters)
            return RandomForestClassifier(**params)

        elif self.model_name == "xgboost":
            # XGBoost - gradient boosting
            if not XGBOOST_AVAILABLE:
                raise ValueError(
                    "XGBoost not installed. Install with: pip install xgboost"
                )
            params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
            }
            if self.model_parameters is not None:
                params.update(self.model_parameters)
            return XGBClassifier(**params)

        elif self.model_name == "ensemble":
            # Voting ensemble of Decision Tree, Random Forest, and XGBoost
            dt = DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
            )
            rf = RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

            estimators = [
                ("decision_tree", dt),
                ("random_forest", rf),
            ]

            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                xgb = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric="logloss",
                )
                estimators.append(("xgboost", xgb))

            # Soft voting (uses predicted probabilities)
            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

        else:
            raise ValueError(f"Model name ({self.model_name}) unknown!")

    def split_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.test_split_size > 0:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                ids_train,
                ids_test,
            ) = train_test_split(
                self.X,
                self.y,
                self.ids,
                test_size=self.test_split_size,
                shuffle=True,
                stratify=self.y,
            )
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        else:
            (
                X_train,
                X_test,
                y_train,
                y_test,
                ids_train,
                ids_test,
            ) = (
                self.X,
                None,
                self.y,
                None,
                self.ids,
                None,
            )
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            ids_train,
            ids_test,
        )

    def validate_model(self, kfold=5):
        # X, _, y, _, _, _ = self.split_data()
        X = self.X
        y = self.y
        groups = self.ids
        kfold = StratifiedGroupKFold(n_splits=kfold)

        models = []
        train_metrics, val_metrics = [], []
        for ith_fold, (train_idx, val_idx) in enumerate(
            kfold.split(X, y, groups=groups)
        ):
            X_train, y_train, X_val, y_val = (
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
            )

            train_ids = set(self.ids[train_idx])
            val_ids = set(self.ids[val_idx])

            overlap = train_ids.intersection(val_ids)
            if len(overlap) > 0:
                print(f"Leakage in fold {ith_fold}: {len(overlap)} overlapping IDs")
            if self.model_name == "rf_grid":
                model = self.rf_grid_search(X_train, y_train)
            elif self.model_name == "xgb_grid":
                model = self.xgb_grid_search(X_train, y_train)
            elif self.model_name == "ensemble":
                model = self.ensemble_model(X_train, y_train)
            else:
                model = self.build_model()
                model.fit(X_train, y_train)
            train_metrics.append(evaluate(model, X_train, y_train))
            val_metrics.append(evaluate(model, X_val, y_val))
            models.append(model)
        train_metrics_summary = pd.DataFrame(train_metrics).mean(numeric_only=True)
        val_metrics_summary = pd.DataFrame(val_metrics).mean(numeric_only=True)

        return train_metrics_summary, val_metrics_summary, models

    # random forest grid search
    # this function performs grid search to find the best hyperparameters for a Random Forest classifier
    # it returns the best estimator found by the grid search
    # the hyperparameters being tuned are n_estimators and max_depth
    # it uses 5-fold cross-validation and f1 score as the scoring metric
    def rf_grid_search(self, X, y):
        param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10]}

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        grid.fit(X, y)
        return grid.best_estimator_

    # xgboost grid search
    # this function performs grid search to find the best hyperparameters for an XGBoost classifier
    # it takes in the feature matrix X and target vector y as inputs
    # it returns the best estimator found by the grid search
    def xgb_grid_search(self, X, y):
        if (len(set(y)) < 2) or (len(y) < 5):
            print("Skipping XGBoost: only one class present")
            return None
        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }
        grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        grid.fit(X, y)
        return grid.best_estimator_

    # this function creates the ensemble model using voting classifier method
    # and then it combines a random forest model and an xgboost model
    # and then it fits the ensemble model on the feature matrix X and target vector y
    def ensemble_model(self, X, y):
        rf_model = self.rf_grid_search(X, y)
        xg_model = self.xgb_grid_search(X, y)

        if xg_model is None:
            return rf_model

        model = VotingClassifier(
            estimators=[("rf", rf_model), ("xgb", xg_model)],
            voting="soft",
        )
        model.fit(X, y)
        return model

    def train(self):
        X, X_test, y, y_test, _, _ = self.split_data()
        if self.model_name == "rf_grid":
            model = self.rf_grid_search(X, y)
        elif self.model_name == "xgb_grid":
            model = self.xgb_grid_search(X, y)
        elif self.model_name == "ensemble":
            model = self.ensemble_model(X, y)
        else:
            model = self.build_model()
            model.fit(X, y)

        train_metrics = evaluate(model, X, y)
        if X_test is not None:
            test_metrics = evaluate(model, X_test, y_test)
        else:
            test_metrics = None

        return model, train_metrics, test_metrics
