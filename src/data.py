"""LHC Olympics 2020 R&D dataset: download, feature construction, splits.

Data source
-----------
Kasieczka, Nachman & Shih, "R&D Dataset for LHC Olympics 2020 Anomaly Detection
Challenge", Zenodo record 6466204 (v5), CC-BY 4.0.
https://doi.org/10.5281/zenodo.6466204

We use ONLY the high-level feature file (74 MB), not the 2.9 GB raw hadron file.
Each row is one event, already clustered into anti-kT R=1 jets by the dataset
authors using fastjet. Columns:

    pxj1 pyj1 pzj1 mj1 tau1j1 tau2j1 tau3j1
    pxj2 pyj2 pzj2 mj2 tau1j2 tau2j2 tau3j2 label

label = 1 for signal (W' -> XY, X->qq, Y->qq; 3.5 TeV / 500 GeV / 100 GeV),
        0 for QCD dijet background.
1M background + 100k signal events, Pythia8 + Delphes 3.4.1, no pileup/MPI,
single fat-jet trigger with pT > 1.2 TeV.
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

RND_URL = ("https://zenodo.org/records/6466204/files/"
           "events_anomalydetection_v2.features.h5?download=1")
RND_MD5 = "271cf5e71fc756b2a8d2b32730689bdb"
RND_NAME = "events_anomalydetection_v2.features.h5"

# Secondary signal (3-prong, X,Y->qqq), used only as a never-trained-on
# generalisation test. 5.2 MB.
QQQ_URL = ("https://zenodo.org/records/6466204/files/"
           "events_anomalydetection_Z_XY_qqq.features.h5?download=1")
QQQ_MD5 = "1e729f7dff225451182c28afaa4bb411"
QQQ_NAME = "events_anomalydetection_Z_XY_qqq.features.h5"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _md5(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dst: str, md5: str, tries: int = 6) -> str:
    """Fetch `url` to `dst`, verifying md5. No-op if already present and valid.

    Zenodo intermittently returns 502/504 under load, so we retry with
    exponential backoff rather than failing the run.
    """
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst) and _md5(dst) == md5:
        print(f"[data] cached, md5 verified: {dst}")
        return dst

    last = None
    for attempt in range(1, tries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=180) as r, open(dst, "wb") as f:
                got = 0
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    got += len(block)
            digest = _md5(dst)
            if digest != md5:
                raise IOError(f"md5 mismatch: got {digest}, expected {md5}")
            print(f"[data] downloaded {got/1e6:.1f} MB in {time.time()-t0:.0f}s, "
                  f"md5 verified: {dst}")
            return dst
        except Exception as exc:                      # noqa: BLE001
            last = exc
            print(f"[data] attempt {attempt}/{tries} failed: "
                  f"{type(exc).__name__}: {exc}")
            if attempt < tries:
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise RuntimeError(
        f"Could not download {url} after {tries} attempts. Last error: {last}. "
        "Zenodo (CERN-hosted) is occasionally down; retry later or fetch the "
        "file manually into the data/ directory."
    )


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

# The six QAE inputs. mjj is deliberately NOT among them: it is the resonance
# variable a real search bump-hunts in, so letting the anomaly score depend on
# it would sculpt the very spectrum the search relies on. Same convention as
# the ANODE / CATHODE / CWoLa-hunting line of work on this dataset.
FEATURES = ["mj1", "dmj", "tau21j1", "tau21j2", "tau32j1", "tau32j2"]


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """n-subjettiness ratio, 0 where the denominator vanishes.

    tau_N = 0 happens for jets with fewer than N reconstructed constituents.
    Those events are physical, not corrupt, so we keep them and map the
    undefined ratio to 0 rather than dropping the event.
    """
    out = np.zeros_like(num, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataset columns -> physics features used in this study.

    Jets are re-ordered by MASS (not pT): j1 is the lighter jet. This is the
    standard convention for this dataset and makes (mj1, dmj) a clean
    parametrisation of the two-jet mass plane, with dmj >= 0 by construction.
    """
    px1, py1, pz1, m1 = (df[c].to_numpy(np.float64)
                         for c in ("pxj1", "pyj1", "pzj1", "mj1"))
    px2, py2, pz2, m2 = (df[c].to_numpy(np.float64)
                         for c in ("pxj2", "pyj2", "pzj2", "mj2"))

    e1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m1**2)
    e2 = np.sqrt(px2**2 + py2**2 + pz2**2 + m2**2)
    mjj = np.sqrt(np.maximum(
        (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2,
        0.0))

    t21_1 = _ratio(df["tau2j1"].to_numpy(np.float64), df["tau1j1"].to_numpy(np.float64))
    t21_2 = _ratio(df["tau2j2"].to_numpy(np.float64), df["tau1j2"].to_numpy(np.float64))
    t32_1 = _ratio(df["tau3j1"].to_numpy(np.float64), df["tau2j1"].to_numpy(np.float64))
    t32_2 = _ratio(df["tau3j2"].to_numpy(np.float64), df["tau2j2"].to_numpy(np.float64))

    # order by mass: light jet first
    light_is_1 = m1 <= m2
    mj_light = np.where(light_is_1, m1, m2)
    mj_heavy = np.where(light_is_1, m2, m1)
    out = pd.DataFrame({
        "mj1": mj_light,
        "dmj": mj_heavy - mj_light,
        "tau21j1": np.where(light_is_1, t21_1, t21_2),
        "tau21j2": np.where(light_is_1, t21_2, t21_1),
        "tau32j1": np.where(light_is_1, t32_1, t32_2),
        "tau32j2": np.where(light_is_1, t32_2, t32_1),
        "mjj": mjj,                       # held out: sculpting diagnostic only
    })
    if "label" in df.columns:
        out["label"] = df["label"].to_numpy(np.int64)
    return out


def load_rnd(data_dir: str = "data") -> pd.DataFrame:
    """Download (if needed) and return the LHCO R&D features with derived columns."""
    path = download(RND_URL, os.path.join(data_dir, RND_NAME), RND_MD5)
    raw = pd.read_hdf(path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"expected a DataFrame from {path}, got {type(raw)}")
    print(f"[data] raw shape {raw.shape}, columns {list(raw.columns)}")
    return build_features(raw)


@dataclass
class Splits:
    """Background-only train/val, plus a mixed test set. All arrays are raw
    (unscaled) feature values; scaling is fit on `train` alone downstream."""
    train: np.ndarray          # background only
    val: np.ndarray            # background only (monitoring / early stopping)
    test: np.ndarray           # background + signal
    test_label: np.ndarray     # 1 = signal
    test_mjj: np.ndarray       # for the sculpting diagnostic
    feature_names: list


def make_splits(df: pd.DataFrame, n_train: int = 100_000, n_val: int = 20_000,
                seed: int = 0) -> Splits:
    """Unsupervised setup: the model only ever sees background during training.

    All remaining background goes into the test set, which maximises the
    background statistics available for measuring rejection at fixed signal
    efficiency (the metric that needs them most).
    """
    rng = np.random.default_rng(seed)
    bkg = df[df.label == 0]
    sig = df[df.label == 1]

    perm = rng.permutation(len(bkg))
    tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    if len(te) == 0:
        raise ValueError("no background left for the test set; reduce n_train/n_val")

    x = FEATURES
    test = np.concatenate([bkg.iloc[te][x].to_numpy(), sig[x].to_numpy()])
    label = np.concatenate([np.zeros(len(te), np.int64), np.ones(len(sig), np.int64)])
    mjj = np.concatenate([bkg.iloc[te]["mjj"].to_numpy(), sig["mjj"].to_numpy()])

    print(f"[data] train {n_train} bkg | val {n_val} bkg | "
          f"test {len(te)} bkg + {len(sig)} sig")
    return Splits(
        train=bkg.iloc[tr][x].to_numpy(),
        val=bkg.iloc[va][x].to_numpy(),
        test=test, test_label=label, test_mjj=mjj, feature_names=list(x),
    )
