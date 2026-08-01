"""Predict the leaderboard score from the one real observation we own.

WHY NOT THE OOV CURVE

`scripts/oov_calibration.py` fits per-clip out-of-vocabulary rate against per-clip
score on 900 dev clips. On phase 1 that works. On phase 2 it reports that 99.7% of
clips fall outside its fitted range and projects 0.15 -- a number produced almost
entirely by extrapolation. It was also measuring the wrong thing: a Luganda model
transcribing Lingala audio emits Luganda-OOV text no matter how good the audio is,
so the 72% OOV was the fingerprint of misrouting, not of degraded audio.

WHAT THIS DOES INSTEAD

One submission has been scored: 0.491944347. Every phase-2 file obeys

    score = a * s + (1 - a) * f

        a  fraction of clips routed to the right language
        s  score on a correctly-routed clip
        f  score on a misrouted clip

f is MEASURED (kaggle/kernels/misroute, a full derangement over the dev set).
a is bounded from the language MIX of the file and the mix each hypothesis claims
phase 2 to be -- see `agreement_bounds`. That leaves s, which the observation pins
down. Then s projects any new file with a known routing.

THE FALSIFICATION

Two routers disagree about what phase 2 is, and they cannot both be right:

    H_mms  facebook/mms-lid-256      lin  0.5%  sna  5.9%  lug 93.5%
    H_ctc  CTC-confidence router     lin 36.9%  sna 15.5%  lug 47.5%

Under each, invert the observation for s. s is a CEILING: what our submitted file
would have scored with perfect routing. Any hypothesis whose ceiling sits below a
score another team has already posted is refuted -- their clips are our clips, and
a ceiling cannot sit under an observed floor. This uses only the public
leaderboard. No labels, no ground truth, nothing rule-restricted.

Usage:
    PYTHONUTF8=1 python -X utf8 scripts/anchor_calibration.py
"""

from __future__ import annotations

import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "zindi" / "Train.csv"
ROUTER_MAP = ROOT / "artifacts" / "router" / "lang_map_asr-conf_z_neg_entropy.json"

SUBMITTED = ROOT / "submissions" / "submission_01_mms_zeroshot_phase2.csv"
OBSERVED = 0.491944347                 # submitted 30 Jul, the only scored file we have
TOP = 0.725666538                      # rank 1; six teams sit within 0.005 of it

CANDIDATE = ROOT / "artifacts" / "lineup" / "submission_03_lineup_lm_phase2.csv"

# MEASURED by kaggle/kernels/misroute: all 900 dev clips through a fixed derangement
# (lin->sna, sna->lug, lug->lin), scored against true references. WER 1.0140 -- every
# word wrong, plus insertions -- but CER only 0.3944, because the wrong model hears the
# same phonemes and writes them in a related Bantu orthography. CER carries the floor.
MISROUTE_F = 0.29581173855876064

LANGS = ("lin", "sna", "lug")

# What each router claims phase 2 IS. H_mms is from the v2 lineup log (mms-lid-256 routed
# lin 8 / sna 89 / lug 1403); H_ctc is the router kernel's winning rule.
H_MMS = {"lin": 8 / 1500, "sna": 89 / 1500, "lug": 1403 / 1500}
H_CTC = {"lin": 0.369, "sna": 0.155, "lug": 0.475}

DEV_MMS_BASELINE = 0.7453   # the checkpoints behind the submitted file
DEV_LINEUP = 0.7985         # the checkpoints behind the candidate, measured v3
ROUTER_ACC_LABELLED = 0.9658

# The candidate's TRUE routing, straight from the v3 kernel log. Not inferred.
KNOWN_CANDIDATE = {"lin": 554 / 1500, "sna": 233 / 1500, "lug": 713 / 1500}


def tokenise(text: str) -> list[str]:
    return [w for w in str(text).lower().split() if w]


class WordLID:
    """Multinomial naive Bayes over word unigrams, add-one smoothed.

    Deliberately simple. The three languages have largely disjoint vocabularies, so
    the job is easy and a cleverer model would only hide its own errors. Its purpose
    is narrow: recover which ASR model wrote a row, from the orthography it left.
    """

    def __init__(self) -> None:
        self.logprior: dict[str, float] = {}
        self.loglik: dict[str, dict[str, float]] = {}
        self.default: dict[str, float] = {}

    def fit(self, texts: list[str], labels: list[str]) -> None:
        counts: dict[str, Counter] = defaultdict(Counter)
        docs: Counter = Counter()
        vocab: set[str] = set()
        for text, lang in zip(texts, labels):
            docs[lang] += 1
            toks = tokenise(text)
            counts[lang].update(toks)
            vocab.update(toks)
        v = len(vocab)
        total = sum(docs.values())
        for lang in counts:
            denom = sum(counts[lang].values()) + v
            self.logprior[lang] = math.log(docs[lang] / total)
            self.loglik[lang] = {w: math.log((c + 1) / denom) for w, c in counts[lang].items()}
            self.default[lang] = math.log(1 / denom)

    def predict(self, text: str) -> str:
        toks = tokenise(text)
        if not toks:
            return "??"
        best, best_score = "??", -math.inf
        for lang in self.logprior:
            s = self.logprior[lang]
            lk, d = self.loglik[lang], self.default[lang]
            for w in toks:
                s += lk.get(w, d)
            if s > best_score:
                best, best_score = lang, s
        return best


def agreement_bounds(file_mix: dict[str, float],
                     truth_mix: dict[str, float]) -> tuple[float, float]:
    """Range of routing accuracies consistent with two language mixes.

    Only the marginals are known -- which clips line up is not. The Frechet bounds
    give the range exactly: at best every clip the file assigns to a language is one
    that truly is that language (sum of minima); at worst the overlap is only what
    the pigeonhole forces (sum of positive parts).

    Using bounds rather than a per-clip map is deliberate. It makes the falsification
    below depend on nothing but the two mixes, so it survives even if a specific
    routing file is regenerated or lost.
    """
    hi = sum(min(file_mix.get(k, 0.0), truth_mix.get(k, 0.0)) for k in LANGS)
    lo = sum(max(0.0, file_mix.get(k, 0.0) + truth_mix.get(k, 0.0) - 1.0) for k in LANGS)
    return lo, hi


def invert(observed: float, a: float, f: float) -> float:
    """Score a file would reach with perfect routing, given its actual routing."""
    return (observed - (1 - a) * f) / a


def main() -> None:
    print("=" * 78)
    print("ANCHORED LEADERBOARD CALIBRATION")
    print("=" * 78)
    print(f"\n  measured misroute floor f = {MISROUTE_F:.4f}   (kaggle/kernels/misroute)")

    train = pd.read_csv(TRAIN, engine="python", on_bad_lines="skip")
    train = train[train["language"].isin(LANGS)].dropna(subset=["transcription", "language"])

    holdout = train.sample(frac=0.15, random_state=1337)
    lid = WordLID()
    lid.fit(*(lambda d: (d["transcription"].tolist(), d["language"].tolist()))(
        train.drop(holdout.index)))
    acc = sum(lid.predict(t) == g
              for t, g in zip(holdout["transcription"], holdout["language"])) / len(holdout)
    print(f"\n[1] text-LID check  holdout={len(holdout):,}  accuracy={acc:.4f}")
    if acc < 0.95:
        print("    !! too weak to attribute rows to models; stopping")
        return
    print("    -> a row's orthography reliably identifies the model that wrote it")

    lid = WordLID()
    lid.fit(train["transcription"].tolist(), train["language"].tolist())

    def mix_of(path: Path, label: str) -> dict[str, float]:
        df = pd.read_csv(path)
        got = Counter(lid.predict(r.Target) for r in df.itertuples())
        n = sum(got.values())
        m = {k: got.get(k, 0) / n for k in LANGS}
        print(f"\n[2] {label}\n    {path.name}  rows={n:,}")
        print("    decoded-by: " + "  ".join(f"{k}={got.get(k, 0):,} ({m[k]:.1%})" for k in LANGS))
        return m

    sub_mix = mix_of(SUBMITTED, f"SUBMITTED -> scored {OBSERVED:.6f}")
    cand_mix = mix_of(CANDIDATE, "CANDIDATE (post-provenance lineup, CTC-routed)")

    # ------------------------------------------------------------------ [3]
    print("\n[3] falsification — which router is telling the truth about phase 2")
    print("    Ceiling = what the SUBMITTED file would score with perfect routing.")
    print("    A hypothesis whose ceiling sits below a posted score is refuted.\n")
    print(f"      {'hypothesis':<10} {'claimed lug':>12} {'a range':>16} {'ceiling':>10}")
    print("      " + "-" * 52)
    ceilings: dict[str, float] = {}
    for name, truth in (("H_ctc", H_CTC), ("H_mms", H_MMS)):
        lo, hi = agreement_bounds(sub_mix, truth)
        # Lowest a gives the highest, most generous ceiling. Refute against that.
        ceil = invert(OBSERVED, max(lo, 1e-9), MISROUTE_F)
        ceilings[name] = ceil
        shown = f"{min(ceil, 1.0):.4f}" + ("+" if ceil > 1 else "")
        print(f"      {name:<10} {truth['lug']:>11.1%} {lo:>7.3f}-{hi:<7.3f} {shown:>10}")

    print(f"\n    posted by others: {TOP:.4f}")
    if ceilings["H_mms"] < TOP <= max(ceilings["H_ctc"], 1.0):
        print("\n    *** H_mms IS REFUTED ***")
        print(f"    If phase 2 really were {H_MMS['lug']:.1%} Luganda, our submitted file was")
        print(f"    already routed {agreement_bounds(sub_mix, H_MMS)[0]:.1%}-"
              f"{agreement_bounds(sub_mix, H_MMS)[1]:.1%} right, and even at the most")
        print(f"    generous end its perfect-routing ceiling is {ceilings['H_mms']:.4f} —")
        print(f"    below {TOP:.4f}, which six teams have posted on these same clips.")
        print("    A ceiling cannot sit under an observed floor.")
        print("\n    This rests only on the two language MIXES, not on any per-clip map,")
        print("    so regenerating a routing file cannot change it. The MMS LID's lug")
        print("    recall of exactly 1.000 on labelled audio is the class-bias tell.")
    else:
        print("\n    Neither hypothesis is refuted by this test — do not act on it.")
        return

    # ------------------------------------------------------------------ [4]
    lo_c, hi_c = agreement_bounds(sub_mix, H_CTC)
    print("\n[4] what our decode actually scores on phase 2, routed correctly")
    print(f"      submitted-file routing accuracy under H_ctc : {lo_c:.3f}-{hi_c:.3f}")
    s_lo, s_hi = invert(OBSERVED, hi_c, MISROUTE_F), invert(OBSERVED, max(lo_c, 1e-9), MISROUTE_F)
    s_lo, s_hi = min(s_lo, s_hi), min(max(s_lo, s_hi), 1.0)
    print(f"      -> s = {s_lo:.4f}-{s_hi:.4f}  (baseline checkpoints, dev {DEV_MMS_BASELINE:.4f})")
    print(f"\n    Phase-2 audio is NOT collapsed. Same checkpoints score {DEV_MMS_BASELINE:.4f} on")
    print(f"    dev and {s_lo:.2f}-{s_hi:.2f} here — an ordinary domain shift, not a broken split.")

    # ------------------------------------------------------------------ [5]
    delta = DEV_LINEUP - DEV_MMS_BASELINE
    lo_k, hi_k = agreement_bounds(cand_mix, H_CTC)
    print(f"\n[5] projecting the CANDIDATE  (dev {DEV_LINEUP:.4f}, {delta:+.4f} over baseline)")
    print("    It was routed BY the surviving router, so its accuracy on phase 2 is that")
    print(f"    router's accuracy — {ROUTER_ACC_LABELLED:.4f} on labelled audio, discounted")
    print("    here because phase 2 is out of domain.\n")
    print(f"      {'router acc':>11} | " + "  ".join(f"xfer {t:.0%}" for t in (0.0, 0.5, 1.0)))
    print("      " + "-" * 44)
    grid: list[float] = []
    for a in (0.85, 0.90, ROUTER_ACC_LABELLED):
        row = []
        for t in (0.0, 0.5, 1.0):
            s = min(1.0, (s_lo + s_hi) / 2 + delta * t)
            row.append(a * s + (1 - a) * MISROUTE_F)
        grid += row
        print(f"      {a:>11.4f} | " + "   ".join(f"{v:.4f}" for v in row))
    print(f"\n    projected range : {min(grid):.4f} - {max(grid):.4f}")
    print(f"    currently on the board : {OBSERVED:.4f}   ({min(grid) - OBSERVED:+.4f} worst case)")
    print(f"    leaders : {TOP:.4f}   (gap {TOP - max(grid):+.4f} at best case)")
    # ------------------------------------------------------------------ [6]
    # The text-LID is 1.0000 on clean transcripts and NOT on ASR output. The candidate
    # gives us a calibration point, because its true routing is known exactly from the
    # kernel log: lin 554 / sna 233 / lug 713. Comparing that against what the text-LID
    # reads back out of the same file measures the bias directly.
    print("\n[6] how much to trust [2] — text-LID bias on ASR output")
    print(f"      {'':<6} {'true routing':>13} {'text-LID reads':>15} {'bias':>8}")
    print("      " + "-" * 46)
    for k in LANGS:
        print(f"      {k:<6} {KNOWN_CANDIDATE[k]:>12.1%} {cand_mix[k]:>14.1%} "
              f"{cand_mix[k] - KNOWN_CANDIDATE[k]:>+8.1%}")
    bias_lug = cand_mix["lug"] - KNOWN_CANDIDATE["lug"]
    print(f"\n    The text-LID over-calls Luganda by {bias_lug:+.1%} on decoded audio. Wrongly-")
    print("    routed output is phonetic soup, and Luganda's short open syllables are what")
    print("    soup most resembles. Clean transcripts do not have this problem, which is why")
    print(f"    [1] reads {acc:.4f}. So [2]'s {sub_mix['lug']:.1%} Luganda is an OVERSTATEMENT of how")
    print("    Luganda-routed the submitted file really was.")
    print("\n    Does that rescue H_mms? Correcting the submitted mix by the same bias:")
    corrected = {k: max(0.0, sub_mix[k] - (cand_mix[k] - KNOWN_CANDIDATE[k])) for k in LANGS}
    tot = sum(corrected.values())
    corrected = {k: v / tot for k, v in corrected.items()}
    lo_m, hi_m = agreement_bounds(corrected, H_MMS)
    ceil_m = invert(OBSERVED, max(lo_m, 1e-9), MISROUTE_F)
    print("      corrected mix: " + "  ".join(f"{k}={corrected[k]:.1%}" for k in LANGS))
    print(f"      a range {lo_m:.3f}-{hi_m:.3f}  ->  ceiling {ceil_m:.4f}  vs posted {TOP:.4f}")
    breakeven = (OBSERVED - MISROUTE_F) / (TOP - MISROUTE_F)
    print(f"\n    H_mms survives only if the submitted file's routing accuracy was below")
    print(f"    {breakeven:.3f}. A file that is majority-Luganda by every measure, against a")
    print(f"    truth claimed to be {H_MMS['lug']:.1%} Luganda, cannot agree that badly.")
    print("    The refutation holds under the correction." if ceil_m < TOP else
          "    !! The refutation does NOT survive the correction — stop and re-examine.")
    print(f"\n    consistency check — candidate mix vs H_ctc: {lo_k:.3f}-{hi_k:.3f}")
    print("    (wide for the same reason: [2] reads output, not routing)")

    json.dump(
        {"misroute_f": MISROUTE_F, "observed": OBSERVED, "s_range": [s_lo, s_hi],
         "projection": [min(grid), max(grid)], "submitted_mix": sub_mix,
         "candidate_mix": cand_mix, "ceilings": ceilings},
        open(ROOT / "artifacts" / "anchor_projection.json", "w"), indent=2)
    print(f"\n    wrote artifacts/anchor_projection.json")


if __name__ == "__main__":
    main()
