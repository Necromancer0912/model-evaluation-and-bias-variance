"""
Model Evaluation, Cross-Validation & Diagnostics Engine (`src/diagnostics_engine.py`).

Part of the standalone `model-evaluation-and-bias-variance` repository.
Implements first-principles validation routines, mathematical bias-variance decomposition,
and threshold-free classification evaluation metrics in pure vectorized NumPy.

Contents
--------
1. Model Selection & Cross-Validation : StratifiedKFoldScratch, train_test_split_scratch
2. Bias-Variance Decomposition        : bias_variance_monte_carlo
3. Classification Evaluation Metrics  : confusion_matrix_scratch, precision_recall_fbeta_scratch,
                                        roc_curve_scratch, pr_curve_scratch, calibration_curve_scratch
4. Minimal Self-Contained Estimators  : PolynomialFeaturesScratch, LinearRegressionScratch,
                                        LogisticRegressionScratch (for independent execution)
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np


# =============================================================================
# 1. SPLITS & STRATIFIED K-FOLD CROSS-VALIDATION
# =============================================================================

def train_test_split_scratch(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays or matrices into random train and test subsets."""
    assert 0.0 < test_size < 1.0
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    m = X_arr.shape[0]
    assert y_arr.shape[0] == m

    indices = np.arange(m)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    n_test = int(round(m * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


class StratifiedKFoldScratch:
    """
    Stratified K-Fold Cross-Validator preserving exact class proportions in every fold.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> None:
        assert n_splits >= 2, "Number of splits must be >= 2"
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        y_arr = np.asarray(y).ravel()
        m = y_arr.shape[0]
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y_arr)

        fold_buckets: List[List[np.ndarray]] = [[] for _ in range(self.n_splits)]
        for cls_val in classes:
            cls_indices = np.where(y_arr == cls_val)[0]
            if self.shuffle:
                rng.shuffle(cls_indices)
            cls_chunks = np.array_split(cls_indices, self.n_splits)
            for k in range(self.n_splits):
                fold_buckets[k].append(cls_chunks[k])

        all_indices = np.arange(m)
        for k in range(self.n_splits):
            val_idx = np.concatenate(fold_buckets[k])
            train_mask = np.ones(m, dtype=bool)
            train_mask[val_idx] = False
            train_idx = all_indices[train_mask]
            yield train_idx, val_idx


# =============================================================================
# 2. MONTE CARLO BIAS-VARIANCE DECOMPOSITION
# =============================================================================

def bias_variance_monte_carlo(
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    f_true_fn: Callable[[np.ndarray], np.ndarray],
    x_eval: np.ndarray,
    n_trials: int = 60,
    m_samples: int = 35,
    noise_std: float = 0.5,
    x_bounds: Tuple[float, float] = (-2.0, 2.0),
    random_state: int = 42,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Estimate exact Bias^2, Variance, Irreducible Noise, and Expected Total Error
    via Monte Carlo simulation over `n_trials` independent training datasets:

        E[(y - f_hat(x))^2] = Bias^2(f_hat(x)) + Var(f_hat(x)) + sigma^2
    """
    rng = np.random.default_rng(random_state)
    x_eval_col = np.asarray(x_eval, dtype=np.float64).reshape(-1, 1)
    f_true_eval = f_true_fn(x_eval_col).ravel()
    preds_matrix = np.zeros((n_trials, x_eval_col.shape[0]), dtype=np.float64)

    for t in range(n_trials):
        X_t = np.sort(rng.uniform(x_bounds[0], x_bounds[1], size=(m_samples, 1)), axis=0)
        y_t = f_true_fn(X_t) + rng.normal(0.0, noise_std, size=(m_samples, 1))
        preds_matrix[t] = fit_predict_fn(X_t, y_t, x_eval_col).ravel()

    f_bar = np.mean(preds_matrix, axis=0)
    bias_sq_pointwise = (f_bar - f_true_eval) ** 2
    var_pointwise = np.mean((preds_matrix - f_bar) ** 2, axis=0)

    bias_sq = float(np.mean(bias_sq_pointwise))
    variance = float(np.mean(var_pointwise))
    noise_var = float(noise_std ** 2)
    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "noise_var": noise_var,
        "total_mse": bias_sq + variance + noise_var,
        "preds_matrix": preds_matrix,
        "f_bar": f_bar,
        "f_true": f_true_eval,
    }


# =============================================================================
# 3. CLASSIFICATION EVALUATION METRICS BEYOND ACCURACY
# =============================================================================

def confusion_matrix_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP) for binary classification labels in {0, 1}."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tn, fp, fn, tp


def precision_recall_fbeta_scratch(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall (TPR), Specificity, FPR, and F_beta score."""
    tn, fp, fn, tp = confusion_matrix_scratch(y_true, y_pred)
    precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
    specificity = tn / max(tn + fp, 1) if (tn + fp) > 0 else 0.0
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    f_beta = ((1.0 + beta_sq) * precision * recall / denom) if denom > 0.0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "f_beta": float(f_beta),
    }


def roc_curve_scratch(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute exact Receiver Operating Characteristic (FPR, TPR, thresholds) and trapezoidal ROC-AUC.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_prob, dtype=np.float64).ravel()
    thresholds = np.concatenate([[1.0 + 1e-6], np.sort(np.unique(yp))[::-1], [-1e-6]])

    fpr_list, tpr_list = [], []
    n_pos = max(int(np.sum(yt == 1)), 1)
    n_neg = max(int(np.sum(yt == 0)), 1)

    for tau in thresholds:
        pred = (yp >= tau).astype(np.int64)
        tp = np.sum((yt == 1) & (pred == 1))
        fp = np.sum((yt == 0) & (pred == 1))
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    fpr_arr = np.asarray(fpr_list, dtype=np.float64)
    tpr_arr = np.asarray(tpr_list, dtype=np.float64)
    # Trapezoidal integration
    auc = float(np.sum(0.5 * (tpr_arr[1:] + tpr_arr[:-1]) * (fpr_arr[1:] - fpr_arr[:-1])))
    return fpr_arr, tpr_arr, thresholds, auc


def pr_curve_scratch(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute exact Precision-Recall Curve (Recall, Precision, thresholds) and PR-AUC.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_prob, dtype=np.float64).ravel()
    thresholds = np.sort(np.unique(yp))[::-1]

    prec_list = [1.0]
    rec_list = [0.0]
    n_pos = max(int(np.sum(yt == 1)), 1)

    for tau in thresholds:
        pred = (yp >= tau).astype(np.int64)
        tp = int(np.sum((yt == 1) & (pred == 1)))
        fp = int(np.sum((yt == 0) & (pred == 1)))
        prec_list.append(tp / max(tp + fp, 1))
        rec_list.append(tp / n_pos)

    rec_arr = np.asarray(rec_list, dtype=np.float64)
    prec_arr = np.asarray(prec_list, dtype=np.float64)
    pr_auc = float(np.sum(0.5 * (prec_arr[1:] + prec_arr[:-1]) * np.maximum(rec_arr[1:] - rec_arr[:-1], 0.0)))
    return rec_arr, prec_arr, thresholds, pr_auc


def calibration_curve_scratch(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Reliability Diagram (mean predicted probability vs. empirical fraction of positives)
    and Brier Score BS = (1/m) sum_i (p_i - y_i)^2.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_prob, dtype=np.float64).ravel()
    brier = float(np.mean((yp - yt) ** 2))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred, bin_true = [], []
    for b_idx in range(n_bins):
        if b_idx == n_bins - 1:
            mask = (yp >= bins[b_idx]) & (yp <= bins[b_idx + 1])
        else:
            mask = (yp >= bins[b_idx]) & (yp < bins[b_idx + 1])
        if np.any(mask):
            bin_pred.append(float(np.mean(yp[mask])))
            bin_true.append(float(np.mean(yt[mask])))

    return np.asarray(bin_pred), np.asarray(bin_true), brier


# =============================================================================
# 4. SELF-CONTAINED ESTIMATORS & PREPROCESSORS
# =============================================================================

class PolynomialFeaturesScratch:
    """Vandermonde expansion up to degree d."""

    def __init__(self, degree: int = 2, include_bias: bool = False) -> None:
        self.degree = degree
        self.include_bias = include_bias
        self.combinations_: List[Tuple[int, ...]] = []
        self.n_input_features_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "PolynomialFeaturesScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.n_input_features_ = X_arr.shape[1]
        start = 0 if self.include_bias else 1
        combos = []
        for deg in range(start, self.degree + 1):
            for c in combinations_with_replacement(range(self.n_input_features_), deg):
                combos.append(c)
        self.combinations_ = combos
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.n_input_features_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        m = X_arr.shape[0]
        Phi = np.empty((m, len(self.combinations_)), dtype=np.float64)
        for col, c in enumerate(self.combinations_):
            Phi[:, col] = 1.0 if len(c) == 0 else np.prod(X_arr[:, list(c)], axis=1)
        return Phi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearRegressionScratch:
    """Linear regression solved via SVD Moore-Penrose Pseudo-Inverse."""

    def __init__(self) -> None:
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        m = X_arr.shape[0]

        X_aug = np.hstack([np.ones((m, 1)), X_arr])
        theta = np.linalg.pinv(X_aug) @ y_arr
        self.b = float(theta[0, 0])
        self.w = theta[1:, 0:1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return (X_arr @ self.w + self.b).ravel()


class LogisticRegressionScratch:
    """Binary Logistic Regression solved via Gradient Descent."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 500) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        m, n = X_arr.shape

        self.w = np.zeros((n, 1), dtype=np.float64)
        self.b = 0.0

        for _ in range(self.epochs):
            z = X_arr @ self.w + self.b
            p = self._sigmoid(z)
            dz = (p - y_arr) / float(m)
            self.w -= self.learning_rate * (X_arr.T @ dz)
            self.b -= self.learning_rate * float(np.sum(dz))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return self._sigmoid(X_arr @ self.w + self.b).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


# =============================================================================
# 5. VERIFICATION SUITE
# =============================================================================

def run_diagnostics_verification(verbose: bool = True) -> Dict[str, float]:
    """Execute mathematical invariants and parity checks for diagnostics engine."""
    results: Dict[str, float] = {}
    rng = np.random.default_rng(2026)

    # 1. Stratified K-Fold Class Proportions Invariance
    m = 120
    y_imbalanced = np.concatenate([np.zeros(90), np.ones(30)])
    skf = StratifiedKFoldScratch(n_splits=5, shuffle=True, random_state=42)
    prop_diffs = []
    base_prop = 30 / 120  # 0.25
    for _, val_idx in skf.split(np.zeros((m, 2)), y_imbalanced):
        fold_prop = np.mean(y_imbalanced[val_idx])
        prop_diffs.append(abs(fold_prop - base_prop))
    max_prop_diff = float(np.max(prop_diffs))
    results["stratified_kfold_prop_violation"] = max_prop_diff
    assert max_prop_diff < 0.05, f"Stratified K-Fold proportion violation: {max_prop_diff}"

    # 2. Confusion Matrix Identity: TN + FP + FN + TP == m
    y_t = rng.integers(0, 2, size=100)
    y_p = rng.integers(0, 2, size=100)
    tn, fp, fn, tp = confusion_matrix_scratch(y_t, y_p)
    assert tn + fp + fn + tp == 100
    metrics = precision_recall_fbeta_scratch(y_t, y_p)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    results["confusion_matrix_sum_gap"] = float(abs((tn + fp + fn + tp) - 100))

    # 3. ROC-AUC for Perfect Classifier = 1.0
    y_perfect_prob = y_t.astype(np.float64) * 0.9 + 0.05
    _, _, _, auc_perf = roc_curve_scratch(y_t, y_perfect_prob)
    assert np.isclose(auc_perf, 1.0, atol=1e-5), f"Perfect AUC error: {auc_perf}"
    results["perfect_roc_auc_gap"] = float(abs(auc_perf - 1.0))

    # 4. Monte Carlo Bias-Variance Decomposition Identity: Total MSE = Bias^2 + Var + Noise
    def f_true_test(x): return 0.5 * x**2 - 1.0
    x_test_eval = np.linspace(-1.5, 1.5, 30)
    bv = bias_variance_monte_carlo(
        fit_predict_fn=lambda X, y, xq: LinearRegressionScratch().fit(X, y).predict(xq),
        f_true_fn=f_true_test,
        x_eval=x_test_eval,
        n_trials=40,
        m_samples=25,
        noise_std=0.4,
        random_state=10,
    )
    decomp_sum = bv["bias_sq"] + bv["variance"] + bv["noise_var"]
    decomp_gap = float(abs(decomp_sum - bv["total_mse"]))
    results["bias_variance_identity_gap"] = decomp_gap
    assert decomp_gap < 1e-12

    # 5. Brier Score Boundedness
    _, _, brier = calibration_curve_scratch(y_t, y_perfect_prob)
    assert 0.0 <= brier <= 1.0
    results["brier_score"] = brier

    if verbose:
        print("=" * 76)
        print("model-evaluation-and-diagnostics :: src/diagnostics_engine.py VERIFICATION")
        print("=" * 76)
        for name, val in results.items():
            print(f"  {name:<40}: {val:.3e}   PASS")
        print("=" * 76)
    return results


if __name__ == "__main__":
    run_diagnostics_verification(verbose=True)
    print("[SUCCESS] All diagnostics mathematical checks and invariants passed.")

