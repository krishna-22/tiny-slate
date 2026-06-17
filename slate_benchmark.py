"""
SLATE: Sparse Lightweight Additive Threshold Ensemble
=====================================================

Author: (benchmark prototype)
License: MIT
"""
from __future__ import annotations

import numpy as np

__all__ = ["SlateShared"]

from slate_shared import SlateShared


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

import subprocess, sys, importlib

for pkg, mod in [("xgboost", "xgboost"), ("interpret", "interpret"),
                 ("imodels", "imodels")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, time, warnings, traceback, json
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, LabelEncoder,
                                   KBinsDiscretizer)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG ---------------------------------- #
RES        = "results_v2.csv"
N_FOLDS    = 3
SEED       = 0
N_JOBS     = -1
N_TRIALS   = 5           # identical random-search budget for every model
TRACKS     = ["default", "tuned"]
MAX_FIT_S  = 600         # skip remaining folds of a (model,dataset,track) if one fit exceeds this
MAX_TRIAL_S = 400        # if 1st tuning trial exceeds this -> fall back to defaults
SMOKE      = bool(int(os.environ.get("SMOKE", "0")))   # tiny validation run

# dataset spec: name -> (openml_id, row_cap_or_None, task, tracks)
DATASETS = {
    # ---- binary (cap=None => FULL dataset) ----
    "adult":           (1590,  None,   "bin",  TRACKS),
    "bank-marketing":  (1461,  None,   "bin",  TRACKS),
    "electricity":     (151,   None,   "bin",  TRACKS),
    "spambase":        (44,    None,   "bin",  TRACKS),
    "phoneme":         (1489,  None,   "bin",  TRACKS),
    "MagicTelescope":  (1120,  None,   "bin",  TRACKS),
    "eeg-eye-state":   (1471,  None,   "bin",  TRACKS),
    "nomao":           (1486,  None,   "bin",  TRACKS),
    "credit-default":  (42477, None,   "bin",  TRACKS),
    "higgs":           (23512, None,   "bin",  TRACKS),   # FULL 98,050
    "churn":           (40701, None,   "bin",  TRACKS),
    "jm1":             (1053,  None,   "bin",  TRACKS),
    # ---- multiclass ----
    "letter":          (6,     None,   "multi", TRACKS),  # 26 classes
    "satimage":        (182,   None,   "multi", TRACKS),  # 6 classes
    "shuttle":         (40685, None,   "multi", TRACKS),  # 7 classes (rare merged)
    "jannis":          (41168, 50000,  "multi", TRACKS),  # 4 classes
}
if SMOKE:
    DATASETS = {"spambase": (44, 2000, "bin", TRACKS),
                "satimage": (182, 2000, "multi", TRACKS)}
    N_FOLDS, N_TRIALS = 2, 2

# per-model training-row caps (slow libraries), {None: no cap}
TRAIN_CAP = {"FIGS": 8000, "RuleFit": 8000, "L1-GAM": 30000,
             "EBM": 150000, "GA2M": 80000}
RNG = np.random.RandomState(SEED)


# --------- L1-GAM: lasso over the SAME threshold-atom dictionary ------- #
class L1ThresholdGAM(BaseEstimator, ClassifierMixin):
    """Sparse additive baseline/ablation: expand x into binary atoms
    1[x_j <= t] on quantile grids, then L1-penalized logistic regression.
    Same hypothesis class as SLATE; selection by global lasso instead of
    greedy budgeted boosting."""
    def __init__(self, n_bins=16, C=0.1, max_cols=6000):
        self.n_bins, self.C, self.max_cols = n_bins, C, max_cols

    def _expand(self, X):
        return self.disc_.transform(X)

    def fit(self, X, y):
        nb = self.n_bins
        while X.shape[1] * nb > self.max_cols and nb > 4:
            nb //= 2
        self.disc_ = Pipeline([
            ("kb", KBinsDiscretizer(n_bins=nb, encode="onehot-dense",
                                    strategy="quantile",
                                    subsample=200000, quantile_method="averaged_inverted_cdf"))])
        self.disc_.fit(X)
        Z = self.disc_.transform(X)
        # cumulative (<= t) coding: cumsum over the one-hot bins per feature
        self.lr_ = LogisticRegression(penalty="l1", C=self.C, solver="saga",
                                      max_iter=300, n_jobs=N_JOBS)
        self.lr_.fit(Z, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._expand(X))

    def predict(self, X):
        return self.lr_.predict(self._expand(X))

    @property
    def n_parameters_(self):
        return int(np.count_nonzero(self.lr_.coef_) + self.lr_.intercept_.size)


# ------------------------- model registry ------------------------------ #
def registry(task):
    import xgboost as xgb
    from interpret.glassbox import ExplainableBoostingClassifier
    from imodels import FIGSClassifier
    try:
        from imodels import RuleFitClassifier
    except ImportError:
        RuleFitClassifier = None
    multi = task == "multi"

    def wrap_ovr(mk):
        return lambda **kw: OneVsRestClassifier(mk(**kw), n_jobs=1)

    R = {}
    # name: (constructor(**params), default_params, search_space, needs_scaling)
    R["SLATE"] = (lambda **kw: SlateShared(random_state=SEED, **kw), {},
                  {"budget": [32, 64, 128, 256], "learning_rate": [0.3, 0.5, 0.8],
                   "l2": [1.0, 2.0, 5.0], "n_bins": [16, 32, 64],
                   "l1": [1e-4, 1e-3, 1e-2]}, False)
    R["LogReg"] = (lambda **kw: LogisticRegression(max_iter=2000, n_jobs=N_JOBS, **kw), {},
                   {"C": [0.001, 0.01, 0.1, 1, 10, 100]}, True)
    R["DecisionTree"] = (lambda **kw: DecisionTreeClassifier(random_state=SEED, **kw),
                         {"max_depth": 4},
                         {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 20, 50],
                          "ccp_alpha": [0.0, 1e-4, 1e-3]}, False)
    R["RandomForest"] = (lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **kw),
                         {"n_estimators": 300},
                         {"n_estimators": [200, 400], "max_features": ["sqrt", 0.3, 0.6],
                          "min_samples_leaf": [1, 5, 20], "max_depth": [None, 12, 20]}, False)
    R["HistGB"] = (lambda **kw: HistGradientBoostingClassifier(random_state=SEED, **kw), {},
                   {"learning_rate": [0.03, 0.1, 0.3], "max_iter": [100, 300, 600],
                    "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.1, 1.0],
                    "min_samples_leaf": [10, 20, 50]}, False)
    R["XGBoost"] = (lambda **kw: xgb.XGBClassifier(tree_method="hist", n_jobs=N_JOBS,
                                                   eval_metric="logloss", random_state=SEED,
                                                   verbosity=0, **kw),
                    {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6},
                    {"n_estimators": [200, 500, 1000], "learning_rate": [0.03, 0.1, 0.3],
                     "max_depth": [4, 6, 8, 10], "subsample": [0.7, 1.0],
                     "colsample_bytree": [0.7, 1.0], "reg_lambda": [1, 5, 10],
                     "min_child_weight": [1, 5, 10]}, False)
    R["EBM"] = (lambda **kw: ExplainableBoostingClassifier(random_state=SEED, n_jobs=N_JOBS,
                                                           interactions=0, **kw), {},
                {"max_bins": [128, 256, 512], "learning_rate": [0.005, 0.01, 0.05],
                 "max_rounds": [2000, 5000], "min_samples_leaf": [2, 10]}, False)
    if not multi:  # GA2M: EBM with pairwise interactions (binary only in interpret)
        R["GA2M"] = (lambda **kw: ExplainableBoostingClassifier(random_state=SEED, n_jobs=N_JOBS,
                                                                **kw),
                     {"interactions": 10},
                     {"interactions": [5, 10, 20], "max_bins": [128, 256],
                      "learning_rate": [0.005, 0.01, 0.05]}, False)
    if RuleFitClassifier is not None:
        base = (lambda **kw: RuleFitClassifier(**kw))
        R["RuleFit"] = ((wrap_ovr(base) if multi else base),
                        {"max_rules": 60},
                        {"max_rules": [30, 60, 120], "tree_size": [3, 4]}, False)
    figs = (lambda **kw: FIGSClassifier(**kw))
    R["FIGS"] = ((wrap_ovr(figs) if multi else figs), {"max_rules": 20},
                 {"max_rules": [12, 20, 40]}, False)
    R["L1-GAM"] = (lambda **kw: L1ThresholdGAM(**kw), {},
                   {"C": [0.01, 0.03, 0.1, 0.3, 1.0], "n_bins": [8, 16, 32]}, False)
    R["DTree-d3"] = (lambda **kw: DecisionTreeClassifier(random_state=SEED, **kw),
                     {"max_depth": 3},
                     {"min_samples_leaf": [1, 5, 20, 50], "ccp_alpha": [0.0, 1e-4, 1e-3]}, False)
    R["DTree-d6"] = (lambda **kw: DecisionTreeClassifier(random_state=SEED, **kw),
                     {"max_depth": 6},
                     {"min_samples_leaf": [1, 5, 20, 50], "ccp_alpha": [0.0, 1e-4, 1e-3]}, False)
    R["RF-20"] = (lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **kw),
                  {"n_estimators": 20, "max_depth": 6},
                  {"max_features": ["sqrt", 0.3, 0.6], "min_samples_leaf": [1, 5, 20],
                   "max_depth": [4, 6]}, False)
    R["RF-40"] = (lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS, **kw),
                  {"n_estimators": 40, "max_depth": 6},
                  {"max_features": ["sqrt", 0.3, 0.6], "min_samples_leaf": [1, 5, 20],
                   "max_depth": [4, 6]}, False)
    R["LinearSVM"] = (lambda **kw: CalibratedClassifierCV(
                          LinearSVC(random_state=SEED, **kw),
                          method="sigmoid", cv=3, ensemble=False),
                      {"C": 1.0},
                      {"C": [0.001, 0.01, 0.1, 1.0, 10.0]}, True)
    R["GaussianNB"] = (lambda **kw: GaussianNB(**kw), {},
                       {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}, False)
    return R


# --------------------------- utilities --------------------------------- #
def load(name, did, cap):
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X, y = d.data, d.target
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=X.index)
    # merge ultra-rare classes (e.g. shuttle) so StratifiedKFold works
    vc = y.value_counts()
    rare = vc[vc < 5 * N_FOLDS].index
    if len(rare) and y.nunique() - len(rare) >= 2:
        keep = ~y.isin(rare)
        X, y = X[keep], y[keep]
        y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)
    if cap and len(X) > cap:
        idx = (X.assign(_y=y).groupby("_y", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), int(np.ceil(cap*len(g)/len(X)))),
                                          random_state=SEED))).index[:cap]
        X, y = X.loc[idx], y.loc[idx]
    return X.reset_index(drop=True), y.reset_index(drop=True).values


def make_pre(X, scale=False):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    steps_num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps_num.append(("sc", StandardScaler()))
    tr = [("num", Pipeline(steps_num), num)]
    if cat:
        tr.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 max_categories=12, sparse_output=False)),
        ]), cat))
    return ColumnTransformer(tr, remainder="drop")


def safe_auc(yte, proba, classes):
    if len(classes) == 2:
        p = proba[:, 1] if proba.ndim == 2 else proba
        return roc_auc_score(yte, p)
    return roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                         labels=classes)


def model_size(name, m):
    try:
        if isinstance(m, OneVsRestClassifier):
            return int(sum(model_size(name, e) for e in m.estimators_))
        if hasattr(m, "n_parameters_"):   return int(m.n_parameters_)
        if name == "LogReg":              return int(m.coef_.size + m.intercept_.size)
        if name == "DecisionTree":        return int(m.tree_.node_count)
        if name == "RandomForest":
            return int(sum(t.tree_.node_count for t in m.estimators_))
        if name == "HistGB":
            return int(sum(p[0].get_n_leaf_nodes() for p in m._predictors))
        if name == "XGBoost":
            return sum(s.count("leaf=") for s in m.get_booster().get_dump())
        if name in ("EBM", "GA2M"):
            return int(sum(np.asarray(t).size for t in m.term_scores_))
        if name in ("FIGS", "RuleFit"):
            return int(getattr(m, "complexity_", getattr(m, "max_rules", -1)))
        if name in ("DTree-d3", "DTree-d6"):
            return int(m.tree_.node_count)
        if name in ("RF-20", "RF-40"):
            return int(sum(t.tree_.node_count for t in m.estimators_))
        if name == "LinearSVM":
            est = m.calibrated_classifiers_[0].estimator
            return int(est.coef_.size + est.intercept_.size)
        if name == "GaussianNB":
            return int(m.theta_.size + m.var_.size + m.class_prior_.size)
    except Exception:
        return -1
    return -1


def sample_params(space, rng):
    return {k: v[rng.randint(len(v))] for k, v in space.items()}


def tune(mk, defaults, space, Ztr, ytr, classes, rng):
    """Equal-budget random search on an inner 80/20 stratified holdout."""
    Zi, Zv, yi, yv = train_test_split(Ztr, ytr, test_size=0.2,
                                      random_state=SEED, stratify=ytr)
    best_p, best_s = dict(defaults), -np.inf
    tried = set()
    for t in range(N_TRIALS):
        p = dict(defaults); p.update(sample_params(space, rng))
        key = json.dumps(p, sort_keys=True, default=str)
        if key in tried:
            continue
        tried.add(key)
        t0 = time.time()
        try:
            m = mk(**p).fit(Zi, yi)
            s = safe_auc(yv, m.predict_proba(Zv), classes)
        except Exception:
            continue
        dt = time.time() - t0
        if s > best_s:
            best_s, best_p = s, p
        if t == 0 and dt > MAX_TRIAL_S:    # too slow to tune -> defaults
            return dict(defaults)
    return best_p


def _timed(m, Z):
    t0 = time.time(); m.predict_proba(Z); return time.time() - t0


# ------------------------------ main ----------------------------------- #
_COLS = ["dataset", "task", "n", "d_enc", "model", "track", "fold",
         "auc", "acc", "fit_s", "lat_us", "size", "params"]


def _eval_dataset(item):
    ds, did, cap, task, tracks = item
    rows = []
    try:
        X, y = load(ds, did, cap)
    except Exception as e:
        print(ds, "LOAD FAIL", repr(e), flush=True); return rows
    classes = np.unique(y)
    print(f"## {ds}: n={len(X)} d={X.shape[1]} classes={len(classes)}", flush=True)
    R = {"SLATE": registry(task)["SLATE"]}
    rng = np.random.RandomState(SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    skip_slow = set()
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for mname, (mk, defaults, space, scale) in R.items():
            for track in tracks:
                if (mname, track) in skip_slow:
                    continue
                try:
                    pre = make_pre(Xtr, scale=scale)
                    Ztr = pre.fit_transform(Xtr).astype(np.float64)
                    Zte = pre.transform(Xte).astype(np.float64)
                    capn = TRAIN_CAP.get(mname)
                    if capn and len(Ztr) > capn:
                        sub = rng.choice(len(Ztr), capn, replace=False)
                        Ztr_f, ytr_f = Ztr[sub], ytr[sub]
                    else:
                        Ztr_f, ytr_f = Ztr, ytr
                    params = dict(defaults)
                    if track == "tuned":
                        params = tune(mk, defaults, space, Ztr_f, ytr_f, classes, rng)
                    m = mk(**params)
                    t0 = time.time(); m.fit(Ztr_f, ytr_f)
                    fit_s = time.time() - t0
                    lat = min(_timed(m, Zte) for _ in range(3)) / len(Zte) * 1e6
                    proba = m.predict_proba(Zte)
                    auc = safe_auc(yte, proba, classes)
                    pred = (classes[np.argmax(proba, 1)] if len(classes) > 2
                            else (proba[:, 1] >= 0.5).astype(int))
                    acc = accuracy_score(yte, pred)
                    row = [ds, task, len(X), Ztr.shape[1], mname, track, fold,
                           auc, acc, fit_s, lat, model_size(mname, m),
                           json.dumps(params, default=str)]
                    if fit_s > MAX_FIT_S:
                        skip_slow.add((mname, track))
                        print(f"   [cap] {ds} {mname}/{track} exceeded {MAX_FIT_S}s; "
                              f"later folds skipped", flush=True)
                except Exception as e:
                    print(ds, mname, track, fold, "FAIL", repr(e), flush=True)
                    row = [ds, task, len(X), -1, mname, track, fold,
                           np.nan, np.nan, np.nan, np.nan, -1, "{}"]
                rows.append(row)
                print(f"{ds:18s} f{fold} {track:7s} {mname:12s} auc={row[7]}", flush=True)
    return rows


def _prep_results():
    if os.path.exists(RES):
        old = pd.read_csv(RES)
        old = old[old.model != "SLATE"]
        old.to_csv(RES, index=False)
    else:
        pd.DataFrame(columns=_COLS).to_csv(RES, index=False)


def _append(rows):
    if rows:
        pd.DataFrame(rows, columns=_COLS).to_csv(RES, mode="a", header=False, index=False)


def run(only=None):
    _prep_results()
    for ds, meta in DATASETS.items():
        if only and ds != only:
            continue
        _append(_eval_dataset((ds, *meta)))
    print("ALL DONE ->", RES, flush=True); analyze()


def run_parallel(workers=None):
    import multiprocessing as mp
    workers = workers or os.cpu_count() or 1
    _prep_results()
    items = [(ds, did, cap, task, tracks)
             for ds, (did, cap, task, tracks) in DATASETS.items()]
    print(f"[parallel] {len(items)} datasets across {workers} workers", flush=True)
    with mp.Pool(workers) as pool:
        for rows in pool.imap_unordered(_eval_dataset, items):
            _append(rows)
    print("ALL DONE ->", RES, flush=True); analyze()


def analyze():
    if not os.path.exists(RES):
        return
    df = pd.read_csv(RES).dropna(subset=["auc"])
    for track in df.track.unique():
        sub = df[df.track == track]
        agg = sub.groupby(["dataset", "model"]).auc.mean().reset_index()
        P = agg.pivot(index="dataset", columns="model", values="auc")
        print(f"\n===== AUC ({track}) =====")
        print(P.round(4).to_string())
        print("mean ranks:", P.rank(axis=1, ascending=False).mean().round(2).to_dict())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "parallel":
        run_parallel(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif len(sys.argv) > 2 and sys.argv[1] == "only":
        run(only=sys.argv[2])
    else:
        run()