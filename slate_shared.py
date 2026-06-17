import numpy as np


def _softmax(F):
    Z = F - F.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


class SlateShared:
    """Multinomial SLATE with ONE shared atom pool across all K classes.

    Model:  F_k(x) = b_k + sum_{a in A} alpha_{a,k} * 1[x_{j_a} <= t_a]
    The atom set A (|A| <= B) is selected ONCE and shared by every class;
    each class keeps its own coefficient row. Storage is therefore
        B*(feat + thr) + B*K coefficients + K intercepts
    instead of one-vs-rest's K independent scorers (K * B atoms).
    Greedy atom selection scores the summed multinomial Newton gain over
    all classes; coefficients are refined by a fully-corrective L1-prox pass.
    """

    def __init__(self, budget=64, n_bins=32, max_iter=None, learning_rate=0.5,
                 l2=2.0, l1=1e-3, corrective_every=5, corrective_passes=2,
                 tol=1e-9, random_state=0):
        self.budget = budget
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l2 = l2
        self.l1 = l1
        self.corrective_every = corrective_every
        self.corrective_passes = corrective_passes
        self.tol = tol
        self.random_state = random_state

    def _bin_features(self, X):
        n, d = X.shape
        self.thresholds_ = []
        codes = np.empty((n, d), dtype=np.int32)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for j in range(d):
            col = X[:, j]
            t = np.unique(np.quantile(col, qs))
            if t.size and t[-1] >= col.max():
                t = t[t < col.max()]
            self.thresholds_.append(t.astype(np.float64))
            codes[:, j] = np.searchsorted(t, col, side="left")
        return codes

    def _hist(self, codes_j, W, nb):
        K = W.shape[1]
        out = np.empty((nb + 1, K))
        for k in range(K):
            out[:, k] = np.bincount(codes_j, weights=W[:, k], minlength=nb + 1)
        return out

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n, d = X.shape
        cls_index = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((n, K))
        Y[np.arange(n), [cls_index[v] for v in y]] = 1.0

        codes = self._bin_features(X)
        nthr = np.array([t.size for t in self.thresholds_])
        max_iter = self.max_iter or min(1200, 6 * self.budget)

        prior = np.clip(Y.mean(0), 1e-6, 1)
        self.intercept_ = np.log(prior)
        F = np.tile(self.intercept_, (n, 1))

        atoms = {}                       # (j,b) -> coef row (K,)
        masks = {}                       # (j,b) -> bool mask
        lr, l2, l1 = self.learning_rate, self.l2, self.l1

        for it in range(max_iter):
            p = _softmax(F)
            g = p - Y
            h = np.maximum(p * (1 - p), 1e-12)

            best_gain, best = -1.0, None
            for j in range(d):
                kk = nthr[j]
                if kk == 0:
                    continue
                Gh = self._hist(codes[:, j], g, kk)
                Hh = self._hist(codes[:, j], h, kk)
                Gc = np.cumsum(Gh, 0)[:kk]          # (kk, K) over {x_j <= t_b}
                Hc = np.cumsum(Hh, 0)[:kk]
                gains = np.sum(Gc * Gc / (Hc + l2), axis=1)   # (kk,)
                b = int(np.argmax(gains))
                if gains[b] > best_gain:
                    best_gain, best = float(gains[b]), (j, b)

            if best is None or best_gain < self.tol:
                break
            key = best
            if key not in atoms and len(atoms) >= self.budget:
                key = self._best_existing(masks, g, h, l2)
                if key is None:
                    break
            if key not in masks:
                masks[key] = codes[:, key[0]] <= key[1]
                atoms[key] = np.zeros(K)
            m = masks[key]
            G = g[m].sum(0); H = h[m].sum(0)
            step = -lr * G / (H + l2)
            atoms[key] += step
            F[m] += step

            p = _softmax(F); g = p - Y
            h = np.maximum(p * (1 - p), 1e-12)
            db = -g.sum(0) / (h.sum(0) + l2)
            self.intercept_ += db; F += db

            if (it + 1) % self.corrective_every == 0:
                F = self._corrective(F, Y, atoms, masks)

        F = self._corrective(F, Y, atoms, masks)
        self._pack(atoms)
        return self

    def _best_existing(self, masks, g, h, l2):
        best_gain, best_key = -1.0, None
        for key, m in masks.items():
            G = g[m].sum(0); H = h[m].sum(0)
            gain = float(np.sum(G * G / (H + l2)))
            if gain > best_gain:
                best_gain, best_key = gain, key
        return best_key

    def _corrective(self, F, Y, atoms, masks):
        l2, l1 = self.l2, self.l1
        for _ in range(self.corrective_passes):
            for key in list(atoms.keys()):
                m = masks[key]
                p = _softmax(F); g = p - Y
                h = np.maximum(p * (1 - p), 1e-12)
                G = g[m].sum(0); H = h[m].sum(0) + l2
                anew = atoms[key] - G / H
                anew = np.sign(anew) * np.maximum(np.abs(anew) - l1 / H, 0.0)
                dF = anew - atoms[key]
                F[m] += dF
                atoms[key] = anew
                if not np.any(anew):
                    del atoms[key]; del masks[key]
            p = _softmax(F); g = p - Y
            h = np.maximum(p * (1 - p), 1e-12)
            db = -g.sum(0) / (h.sum(0) + l2)
            self.intercept_ += db; F += db
        return F

    def _pack(self, atoms):
        keys = sorted(atoms.keys())
        self.atom_feature_ = np.array([k[0] for k in keys], dtype=np.int32)
        self.atom_threshold_ = np.array(
            [self.thresholds_[k[0]][k[1]] for k in keys], dtype=np.float64)
        self.atom_coef_ = (np.array([atoms[k] for k in keys], dtype=np.float64)
                           if keys else np.zeros((0, len(self.classes_))))
        self.n_atoms_ = len(keys)
        del self.thresholds_

    def decision_function(self, X):
        X = np.asarray(X, float)
        F = np.tile(self.intercept_, (X.shape[0], 1))
        for a in range(self.n_atoms_):
            jf = self.atom_feature_[a]; t = self.atom_threshold_[a]
            F += np.outer((X[:, jf] <= t).astype(float), self.atom_coef_[a])
        return F

    def predict_proba(self, X):
        return _softmax(self.decision_function(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), 1)]

    @property
    def n_parameters_(self):
        K = len(self.classes_)
        return int(2 * self.n_atoms_ + self.n_atoms_ * K + K)

    @property
    def memory_bytes_(self):
        return int(self.footprint_bytes())

    def footprint_bytes(self):
        B, K = self.n_atoms_, len(self.classes_)
        return 8 * B + 4 * B * K + 4 * K
