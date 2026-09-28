import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, ShuffleSplit, LeaveOneGroupOut

mne.set_log_level("WARNING")

SUBJECTS = list(range(1, 11))   # 10 subjects keeps runtime short
RUNS = [4, 8, 12]               # imagined left vs right fist

def load_subject(s):
    files = eegbci.load_data(s, RUNS)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    raw.filter(8.0, 30.0, skip_by_annotation="edge")  # mu/beta band
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, dict(left=2, right=3), tmin=1.0, tmax=4.0,
                        picks=picks, baseline=None, preload=True)
    return epochs.get_data(), epochs.events[:, -1] - 2

X, y, groups = [], [], []
print("Loading PhysioNet EEGMMI Dataset...")
for s in SUBJECTS:
    Xs, ys = load_subject(s)
    X.append(Xs)
    y.append(ys)
    groups.append(np.full(len(ys), s))
    print(f"Subject {s}: {len(ys)} trials loaded.")

X = np.concatenate(X)
y = np.concatenate(y)
groups = np.concatenate(groups)

# CSP + LDA Pipeline
clf = make_pipeline(
    CSP(n_components=4, reg="ledoit_wolf", log=True),
    LinearDiscriminantAnalysis()
)

print("\nRunning Within-Subject Evaluation...")
within = {}
for s in SUBJECTS:
    m = groups == s
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    within[s] = cross_val_score(clf, X[m], y[m], cv=cv).mean()

print("Running Leave-One-Subject-Out (LOSO) Evaluation...")
loso_scores = cross_val_score(clf, X, y, groups=groups, cv=LeaveOneGroupOut())
loso = dict(zip(SUBJECTS, loso_scores))

# Save & Display Results
df = pd.DataFrame({
    "subject": SUBJECTS,
    "within_subject": [within[s] for s in SUBJECTS],
    "leave_one_subject_out": [loso[s] for s in SUBJECTS]
})

df.to_csv("reports/results.csv", index=False)

print("\n--- Evaluation Summary ---")
print(df.round(3))
print("\n--- Means ---")
print(df[["within_subject", "leave_one_subject_out"]].mean().round(3))

# Plot Comparison
ax = df.plot(x="subject", y=["within_subject", "leave_one_subject_out"], kind="bar", figsize=(9, 4))
plt.axhline(0.5, ls="--", c="gray", label="Chance (0.50)")
plt.ylabel("Accuracy")
plt.title("Motor Imagery (Left vs Right Fist): Within-Subject vs LOSO Generalization")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("reports/accuracy_comparison.png", dpi=150)
print("\nPlot saved to 'reports/accuracy_comparison.png'")
