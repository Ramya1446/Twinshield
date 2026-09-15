"""
Residual anomaly gate -- the channel that does not depend on known attacks.

Why this exists
---------------
Every baseline here is a supervised classifier over raw CAN traffic, and a
supervised classifier is bounded by its label set.  Show it an attack class that
was absent from training and it has no output neuron to land on, so it must call
that traffic something it has seen -- in practice "Normal".  That is the
zero-day failure mode, and no amount of extra CNN or LSTM capacity fixes it,
because the missing ingredient is not capacity but a reference for what the bus
*should* be doing.

TwinShield-Fusion has that reference.  The GRU Digital Twin is fitted on normal
traffic only, so the residual |actual - expected| measures deviation from
learnt-normal behaviour and is defined for any frame whatsoever -- including one
belonging to an attack nobody has ever labelled.  This gate turns that residual
into a decision.

Two views of the same residual
------------------------------
**Frame view** -- how far this one frame departs from what its own ECU normally
looks like, per payload byte:

    z_frame(t) = max over bytes b of   (r[t,b] - mu[i,b]) / sigma[i,b]

where i is the frame's arbitration ID.  This is the sharp channel: a forged
payload that moves one byte from ~0.5 to ~58 shows up immediately, and it does
not get averaged away by the surrounding traffic.

**ECU view** -- how far a whole ECU stream has drifted across the 32-frame
context window:

    z_ecu(t)   = max over streams s present in the window of
                 (mean residual of s in the window - mu_s) / sigma_s

This is the sustained channel: it survives a single quiet frame and reflects the
multi-ECU structure the fusion branch is built on.

    score(t) = max(z_frame(t), z_ecu(t))       alarm = score > threshold

Calibration, and why it is the whole argument
---------------------------------------------
`mu` and `sigma` are estimated on normal *training* frames; the threshold is the
(1 - target_fpr) quantile of the score over normal *validation* frames.  No
attack frame of any class takes part in fitting or calibration.  That is what
lets the gate be claimed as attack-agnostic: there is no signature in it to
memorise, only a model of normal.

Per-ID, per-byte standardisation matters.  Some ECUs and some bytes (counters,
noisy sensors) are intrinsically hard for the twin to predict and sit at a high
baseline residual; one global threshold would fire on them constantly.  Judging
each byte of each ECU against its own normal behaviour is what makes a single
bus-wide threshold usable at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Residuals live on the payload's [0, 1] scale, so this floor is one byte of
# standard deviation.  Below that the twin is not resolving anything real, and
# without a floor a byte the twin happens to predict perfectly would turn
# ordinary jitter into an enormous z-score.
SIGMA_FLOOR = 1.0 / 255.0


def _per_slot_moments(values: np.ndarray, slot: np.ndarray, n_slots: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of `values` [n, dim], grouped by `slot` [n]. -> two [n_slots, dim]."""
    dim = values.shape[1]
    mu = np.zeros((n_slots, dim), dtype=np.float64)
    sigma = np.full((n_slots, dim), SIGMA_FLOOR, dtype=np.float64)
    counts = np.bincount(slot, minlength=n_slots).astype(np.float64)
    ok = counts > 1

    sums = np.zeros((n_slots, dim), dtype=np.float64)
    np.add.at(sums, slot, values)
    mu[ok] = sums[ok] / counts[ok, None]

    sq = np.zeros((n_slots, dim), dtype=np.float64)
    np.add.at(sq, slot, (values - mu[slot]) ** 2)
    sigma[ok] = np.sqrt(sq[ok] / counts[ok, None])
    # A slot seen once or never keeps the floor: no evidence is not low variance.
    return mu, np.maximum(sigma, SIGMA_FLOOR)


@dataclass
class ResidualAnomalyGate:
    mu_frame: np.ndarray       # [S, 8] normal per-ID per-byte residual mean
    sigma_frame: np.ndarray    # [S, 8]
    mu_ecu: np.ndarray         # [S] normal windowed per-ECU residual mean
    sigma_ecu: np.ndarray      # [S]
    threshold: float           # score above which the gate raises an alarm
    target_fpr: float          # FPR the threshold was calibrated to
    n_fit_frames: int = 0      # normal frames behind mu / sigma
    n_cal_frames: int = 0      # normal frames behind the threshold

    # -- the two channels ---------------------------------------------------
    def frame_z(self, residual: np.ndarray, id_index: np.ndarray) -> np.ndarray:
        """Per-byte deviation of each frame from its own ECU's normal. [N, 8]"""
        return (residual - self.mu_frame[id_index]) / self.sigma_frame[id_index]

    def ecu_z(self, streams: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Per-stream deviation across the window; absent streams are -inf. [N, S]"""
        v = streams.mean(axis=2)
        z = (v - self.mu_ecu[None, :]) / self.sigma_ecu[None, :]
        return np.where(mask > 0, z, -np.inf)

    # -- scoring ------------------------------------------------------------
    def score(self, residual: np.ndarray, id_index: np.ndarray,
              streams: np.ndarray, mask: np.ndarray) -> np.ndarray:
        zf = self.frame_z(residual, id_index).max(axis=1)
        ze = self.ecu_z(streams, mask).max(axis=1)
        # A window with no stream at all cannot be judged on the ECU channel;
        # fall back to the frame channel rather than alarming on missing evidence.
        ze = np.where(np.isfinite(ze), ze, -np.inf)
        return np.maximum(zf, ze).astype(np.float32)

    def explain(self, residual: np.ndarray, id_index: np.ndarray,
                streams: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Score plus the parts the UI shows: which channel, byte and ECU fired."""
        zf_all = self.frame_z(residual, id_index)
        zf = zf_all.max(axis=1)
        ze_all = self.ecu_z(streams, mask)
        ze = np.where(np.isfinite(ze_all.max(axis=1)), ze_all.max(axis=1), -np.inf)
        return {
            "score": np.maximum(zf, ze).astype(np.float32),
            "frame_z": zf.astype(np.float32),
            "ecu_z": np.where(np.isfinite(ze), ze, 0.0).astype(np.float32),
            "top_byte": zf_all.argmax(axis=1),
            "top_stream": np.where(np.isfinite(ze_all), ze_all,
                                   -np.finfo(np.float32).max).argmax(axis=1),
            "channel": np.where(zf >= ze, "frame", "ecu"),
        }

    def alarm(self, residual: np.ndarray, id_index: np.ndarray,
              streams: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.score(residual, id_index, streams, mask) > self.threshold

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, residual: np.ndarray, id_index: np.ndarray,
            streams: np.ndarray, mask: np.ndarray,
            fit_idx: np.ndarray, cal_idx: np.ndarray,
            target_fpr: float = 0.01) -> "ResidualAnomalyGate":
        """Fit on normal training frames, calibrate on normal validation frames.

        `fit_idx` and `cal_idx` must contain **only Normal frames**; the caller
        guarantees that, and it is the property the whole zero-day claim rests on.

        The two index sets must also be *different* frames, and `fit_idx` is
        expected to be the twin's own training frames.  The twin predicts those
        better than it predicts anything else, so mu and sigma estimated there
        are optimistic and a threshold read off the same frames would understate
        the false-alarm rate.  Calibrating on held-out normal frames against
        those same mu and sigma is what makes the target FPR mean something.
        """
        n_slots = streams.shape[1]
        mu_f, sg_f = _per_slot_moments(residual[fit_idx], id_index[fit_idx], n_slots)

        v = streams[fit_idx].mean(axis=2)                     # [n, S]
        m = mask[fit_idx] > 0
        counts = m.sum(axis=0).astype(np.float64)
        mu_e = np.zeros(n_slots, dtype=np.float64)
        sg_e = np.full(n_slots, SIGMA_FLOOR, dtype=np.float64)
        ok = counts > 1
        # A stream absent from a window says nothing about that stream and must
        # not be averaged in as a zero.
        mu_e[ok] = np.where(m, v, 0.0).sum(axis=0)[ok] / counts[ok]
        dev = np.where(m, v - mu_e[None, :], 0.0)
        sg_e[ok] = np.sqrt((dev ** 2).sum(axis=0)[ok] / counts[ok])
        sg_e = np.maximum(sg_e, SIGMA_FLOOR)

        gate = cls(mu_frame=mu_f.astype(np.float32), sigma_frame=sg_f.astype(np.float32),
                   mu_ecu=mu_e.astype(np.float32), sigma_ecu=sg_e.astype(np.float32),
                   threshold=0.0, target_fpr=float(target_fpr),
                   n_fit_frames=int(len(fit_idx)), n_cal_frames=int(len(cal_idx)))

        cal = gate.score(residual[cal_idx], id_index[cal_idx],
                         streams[cal_idx], mask[cal_idx])
        gate.threshold = float(np.quantile(cal, 1.0 - target_fpr))
        return gate

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        np.savez(path, mu_frame=self.mu_frame, sigma_frame=self.sigma_frame,
                 mu_ecu=self.mu_ecu, sigma_ecu=self.sigma_ecu,
                 threshold=np.float32(self.threshold),
                 target_fpr=np.float32(self.target_fpr),
                 n_fit_frames=np.int64(self.n_fit_frames),
                 n_cal_frames=np.int64(self.n_cal_frames))
        return path

    @classmethod
    def load(cls, path: str) -> "ResidualAnomalyGate":
        z = np.load(path)
        return cls(mu_frame=z["mu_frame"], sigma_frame=z["sigma_frame"],
                   mu_ecu=z["mu_ecu"], sigma_ecu=z["sigma_ecu"],
                   threshold=float(z["threshold"]), target_fpr=float(z["target_fpr"]),
                   n_fit_frames=int(z["n_fit_frames"]), n_cal_frames=int(z["n_cal_frames"]))

    def describe(self) -> Dict[str, float]:
        return {"threshold": round(float(self.threshold), 4),
                "target_fpr": self.target_fpr,
                "n_fit_frames": self.n_fit_frames,
                "n_cal_frames": self.n_cal_frames}


def sweep(score: np.ndarray, is_normal: np.ndarray, groups: Dict[str, np.ndarray],
          n_points: int = 80) -> Dict[str, list]:
    """Detection rate against false-positive rate as the threshold is swept.

    The Results page plots this so the operating point actually used can be read
    against the alternatives, rather than presented as the only possible choice.
    """
    finite = score[np.isfinite(score)]
    if len(finite) == 0:
        return {"threshold": [], "fpr": []}
    ths = np.linspace(float(finite.min()), float(finite.max()), n_points)
    out: Dict[str, list] = {"threshold": [float(t) for t in ths]}
    out["fpr"] = [float((score[is_normal] > t).mean()) if is_normal.any() else 0.0
                  for t in ths]
    for name, sel in groups.items():
        out[name] = [float((score[sel] > t).mean()) if sel.any() else 0.0 for t in ths]
    return out
