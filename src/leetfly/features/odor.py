"""Turn problem text into a "smell": activity of K receptor types.

Deliberately dumb and unsupervised: TF-IDF -> NMF. Each NMF component plays the role of one odorant-receptor
type, and a problem's receptor activity is its non-negative mixture over those components. The featurizer never sees
labels and is fit on training problems only.

Everything here is re-implemented in web/src/fly.ts, so the tokenizer and the NMF transform are hand-written and
deterministic (no sklearn internals that the browser can't reproduce).
"""

import re

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

TOKEN = re.compile(r"[a-z0-9^]+")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
NMF_ITERS = 200
EPS = 1e-9


def analyze(text: str) -> list[str]:
    """Lowercase word tokens (>= 2 chars, not stop words, not bare numbers) plus adjacent bigrams."""
    words = [
        t for t in TOKEN.findall(text.lower()) if len(t) >= 2 and t not in STOP_WORDS and not t.isdigit()
    ]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def nmf_transform(x: sparse.spmatrix | np.ndarray, h: np.ndarray, iters: int = NMF_ITERS) -> np.ndarray:
    """Non-negative codes W minimizing ||X - W H||, via a fixed number of multiplicative updates."""
    xht = np.asarray(x @ h.T)  # (n, K)
    hht = h @ h.T  # (K, K)
    w = np.maximum(xht / np.diag(hht), 1e-6)
    for _ in range(iters):
        w *= xht / (w @ hht + EPS)
    return w


class OdorFeaturizer:
    def __init__(self, n_receptors: int, max_terms: int = 8000, min_df: int = 3, seed: int = 0):
        self.n_receptors = n_receptors
        self.tfidf = TfidfVectorizer(
            analyzer=analyze, min_df=min_df, max_features=max_terms, sublinear_tf=True, norm="l2", dtype=np.float64
        )
        self.seed = seed

    def fit(self, texts: list[str]) -> "OdorFeaturizer":
        x = self.tfidf.fit_transform(texts)
        nmf = NMF(n_components=self.n_receptors, init="nndsvda", max_iter=1000, random_state=self.seed)
        nmf.fit(x)
        self.components = nmf.components_  # (K, V)
        r = nmf_transform(x, self.components)
        self.scale = r.mean(axis=0) + EPS  # each receptor's mean activity over training problems = 1
        return self

    def tfidf_matrix(self, texts: list[str]) -> sparse.csr_matrix:
        return self.tfidf.transform(texts)

    def receptors(self, texts: list[str]) -> np.ndarray:
        return nmf_transform(self.tfidf.transform(texts), self.components) / self.scale

    def top_terms(self, n: int = 6) -> list[list[str]]:
        vocab = np.array(self.tfidf.get_feature_names_out())
        return [list(vocab[np.argsort(-row)[:n]]) for row in self.components]
