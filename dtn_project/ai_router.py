"""AI routing module: Random Forest estimates the probability that a candidate
carrier will eventually deliver a message to its destination.

IMPORTANT: no real disaster-network trace data is available, so the training set
is SYNTHETIC (simulated). It is generated from a hand-designed noisy logistic
relationship between the features and delivery success. This is clearly a
prototype assumption, documented in the README and shown in the dashboard.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [
    "distance",        # candidate's distance to destination, normalised 0..1
    "battery",         # candidate battery, 0..1
    "buffer_free",     # fraction of candidate's buffer still free, 0..1
    "encounter_freq",  # how often candidate meets the destination / others, 0..1
    "delivery_rate",   # candidate's past delivery success rate, 0..1
    "hop_count",       # hops the message would have travelled, 0..8+
]

FEATURE_LABELS = {
    "distance": "Distance to destination",
    "battery": "Battery level",
    "buffer_free": "Buffer availability",
    "encounter_freq": "Encounter frequency",
    "delivery_rate": "Previous delivery rate",
    "hop_count": "Hop count",
}


def generate_synthetic_data(n_samples=4000, seed=42):
    """Create a SIMULATED training dataset (not real-world data)."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0, 1, n_samples)
    battery = rng.uniform(0.05, 1, n_samples)
    buffer_free = rng.uniform(0, 1, n_samples)
    encounter_freq = rng.beta(1.2, 2.5, n_samples)
    delivery_rate = rng.beta(2, 2, n_samples)
    hop_count = rng.integers(0, 9, n_samples)

    logit = (
        3.2 * (1 - distance)
        + 1.2 * battery
        + 1.0 * buffer_free
        + 3.6 * encounter_freq
        + 2.4 * delivery_rate
        - 0.30 * hop_count
        - 4.6
        + rng.normal(0, 0.45, n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    delivered = (rng.uniform(0, 1, n_samples) < prob).astype(int)

    return pd.DataFrame({
        "distance": distance, "battery": battery, "buffer_free": buffer_free,
        "encounter_freq": encounter_freq, "delivery_rate": delivery_rate,
        "hop_count": hop_count, "delivered": delivered,
    })


class AIRouter:
    """Random Forest delivery-probability estimator."""

    def __init__(self, n_samples=4000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.model = None
        self.accuracy = None
        self.auc = None
        self.importances = None
        self.dataset = None
        self.is_simulated = True  # always True: training data is synthetic

    def train(self):
        df = generate_synthetic_data(self.n_samples, self.seed)
        X = df[FEATURES].values
        y = df["delivered"].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=self.seed, stratify=y
        )
        self.model = RandomForestClassifier(
            n_estimators=40, max_depth=8, min_samples_leaf=5,
            random_state=self.seed, n_jobs=1,
        )
        self.model.fit(X_tr, y_tr)
        proba = self.model.predict_proba(X_te)[:, 1]
        self.accuracy = float(accuracy_score(y_te, proba > 0.5))
        self.auc = float(roc_auc_score(y_te, proba))
        self.importances = pd.Series(
            self.model.feature_importances_, index=FEATURES
        ).sort_values()
        self.dataset = df
        return self

    def predict_proba(self, X):
        """Return P(delivery) for each row of X (shape n x 6)."""
        X = np.asarray(X, dtype=float).reshape(-1, len(FEATURES))
        return self.model.predict_proba(X)[:, 1]

    def predict_one(self, distance, battery, buffer_free, encounter_freq, delivery_rate, hop_count):
        return float(self.predict_proba([[distance, battery, buffer_free,
                                          encounter_freq, delivery_rate, hop_count]])[0])
