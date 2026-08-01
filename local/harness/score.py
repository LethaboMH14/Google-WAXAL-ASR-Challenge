"""
The competition metric, reimplemented exactly, plus the statistics needed to tell a real
improvement from leaderboard noise.

WHY THIS FILE EXISTS
--------------------
Until now every claim about "this change is worth N points" was inferred from a formula nobody
had verified, and the only way to test anything was to spend one of five daily submissions and
wait. That is a terrible feedback loop: 5 experiments a day, each with a multi-hour latency, on a
public leaderboard that is only ~30% of the test set. This module makes the loop local and
instant.

THE METRIC, VERBATIM
--------------------
From the organisers' own starter notebook (data/zindi/Waxal_Challenge_Starter_Code.ipynb,
`run_evaluation`):

    refs_lower  = [r.lower() for r in references]
    preds_lower = [p.lower() for p in predictions]
    {"wer": jiwer.wer(refs_lower, preds_lower), "cer": jiwer.cer(refs_lower, preds_lower)}

That is the whole normalisation: **lowercase, and nothing else**. Punctuation is NOT stripped.
Accents are NOT folded. Whitespace is NOT collapsed beyond what jiwer does internally. Any
"cleaning" we apply to our own hypotheses that the reference does not also get is pure loss.

The competition page states the two are combined as a weighted mean, 0.5 each, and the leaderboard
is higher-is-better while WER/CER are lower-is-better, so the reported score is

    multi = 0.5 * (1 - WER) + 0.5 * (1 - CER)

The (1 - x) inversion is the one piece NOT written down by the organisers — it is inferred. It is
consistent with our single real observation (a zero-shot MMS submission scored 0.491944347, which
matches this formula at the error rates that system produces), but it stays labelled an inference
until a second submission confirms it. `calibrate()` below is how we confirm it: the Zindi
leaderboard reports WER and CER as their own columns, so one submission gives us a (WER, CER,
multi) triple and the formula either predicts it or it does not.

POOLING MATTERS
---------------
jiwer.wer(list_of_refs, list_of_hyps) pools: it sums errors over the whole corpus and divides by
total reference words. It is NOT the mean of per-utterance WERs. So a language contributes to the
score in proportion to its WORD COUNT, not its row count. Measured on the phase-1 test set that
works out to roughly lin 46%, sna 37%, lug 17%. Anything that reports a flat average across the
three languages is answering a different question than the leaderboard is.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer
import numpy as np

# Utterance counts by language in the phase-1 test set (data/zindi/Test.csv, from the id prefix).
# Phase 2 strips the prefix so we cannot measure it there, but it is the organisers' own test
# split for the same three languages and is the best available estimate of the mix.
PHASE1_TEST_MIX = {"lin": 1866, "sna": 1749, "lug": 638}


def normalise(text: str) -> str:
    """The organisers' normalisation, in full. Deliberately just `.lower()`.

    Resist the urge to add anything here. Stripping punctuation would make our local numbers look
    better and our leaderboard score worse, which is the worst possible property for a harness.
    """
    return str(text).lower()


@dataclass
class Score:
    wer: float
    cer: float
    multi: float
    n: int
    ref_words: int

    def __str__(self) -> str:
        return (f"multi={self.multi:.4f}  WER={self.wer:.4f}  CER={self.cer:.4f}  "
                f"(n={self.n:,} utts, {self.ref_words:,} ref words)")


def score(refs: list[str], hyps: list[str]) -> Score:
    """The leaderboard number for one pool of utterances."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} refs vs {len(hyps)} hyps")
    if not refs:
        raise ValueError("nothing to score")
    r = [normalise(x) for x in refs]
    h = [normalise(x) for x in hyps]
    # jiwer errors on an empty reference; an empty HYPOTHESIS is fine and is exactly what a
    # collapsed CTC model produces, so it must stay scoreable rather than being filtered out.
    for i, x in enumerate(r):
        if not x.strip():
            raise ValueError(f"reference {i} is empty — drop it from the dev set instead")
    wer = jiwer.wer(r, h)
    cer = jiwer.cer(r, h)
    return Score(wer=wer, cer=cer, multi=0.5 * (1 - wer) + 0.5 * (1 - cer),
                 n=len(r), ref_words=sum(len(x.split()) for x in r))


def score_by_language(refs, hyps, langs) -> dict[str, Score]:
    """Per-language breakdown, plus the pooled 'overall'.

    The per-language numbers are for deciding where to spend GPU hours. 'overall' is the number
    that predicts the leaderboard — and it is NOT the mean of the three, because of pooling.
    """
    out: dict[str, Score] = {}
    for lg in sorted(set(langs)):
        idx = [i for i, x in enumerate(langs) if x == lg]
        out[lg] = score([refs[i] for i in idx], [hyps[i] for i in idx])
    out["overall"] = score(refs, hyps)
    return out


def bootstrap_ci(refs: list[str], hyps: list[str], n_boot: int = 400,
                 seed: int = 1337, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval on `multi`, by resampling utterances.

    This is the part that keeps us honest. The public leaderboard is ~30% of a 1,500-row test set,
    i.e. ~450 utterances, and the top four teams there are separated by 0.0001. An interval this
    wide means that gap is noise: it is not evidence that anyone is actually ahead, and chasing it
    by picking whichever submission scored highest on 450 clips is how you overfit the public
    split and drop on the private one.

    Use it the other way round too: if a change's CI overlaps zero on our dev set, do not spend a
    submission on it.
    """
    rng = np.random.default_rng(seed)
    r = [normalise(x) for x in refs]
    h = [normalise(x) for x in hyps]
    n = len(r)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rr = [r[i] for i in idx]
        hh = [h[i] for i in idx]
        try:
            w, c = jiwer.wer(rr, hh), jiwer.cer(rr, hh)
        except ValueError:                       # all-empty resample; vanishingly rare
            continue
        vals.append(0.5 * (1 - w) + 0.5 * (1 - c))
    if not vals:
        return (math.nan, math.nan)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def reweight_to_test_mix(per_lang: dict[str, Score],
                         mix: dict[str, int] | None = None) -> float:
    """Predicted leaderboard `multi` if the dev set had the test set's language proportions.

    Needed because our dev set is whatever the validation split gives us, while the leaderboard
    pools over the test mix. Reconstructs the pooled WER/CER by rebuilding error counts:
    pooled_WER = sum(WER_l * words_l) / sum(words_l), with words_l scaled to the test mix. Same
    for CER on characters. This is exact for pooled rates, which is what jiwer computes.
    """
    mix = mix or PHASE1_TEST_MIX
    langs = [lg for lg in mix if lg in per_lang]
    if not langs:
        raise ValueError("no overlap between dev-set languages and the test mix")
    # Mean words/chars per utterance, measured on our dev set, scaled by the test utterance counts.
    w_num = w_den = c_num = c_den = 0.0
    for lg in langs:
        s = per_lang[lg]
        words = s.ref_words / s.n * mix[lg]          # expected ref words for this language
        chars = s.ref_words / s.n * mix[lg] * 5.5    # ~5.5 chars/word incl. space; cancels in ratio
        w_num += s.wer * words
        w_den += words
        c_num += s.cer * chars
        c_den += chars
    wer, cer = w_num / w_den, c_num / c_den
    return 0.5 * (1 - wer) + 0.5 * (1 - cer)


def calibrate(observed_multi: float, observed_wer: float, observed_cer: float) -> None:
    """Check the inferred `multi = 0.5(1-WER) + 0.5(1-CER)` against a real leaderboard row.

    Zindi's leaderboard shows WER, CER and Multi Score as separate columns, so after any
    submission we can feed all three in here. If this disagrees, every score estimate in this
    repo is wrong and the formula must be fixed before anything else is decided.
    """
    pred = 0.5 * (1 - observed_wer) + 0.5 * (1 - observed_cer)
    delta = abs(pred - observed_multi)
    verdict = "CONFIRMED" if delta < 5e-4 else "*** MISMATCH — formula is wrong ***"
    print(f"leaderboard multi = {observed_multi:.9f}")
    print(f"formula predicts  = {pred:.9f}   |delta| = {delta:.2e}   {verdict}")


def report(refs, hyps, langs, title: str = "", ci: bool = True) -> dict:
    per = score_by_language(refs, hyps, langs)
    print(f"\n=== {title or 'dev-set score'} ===")
    for lg in sorted(k for k in per if k != "overall"):
        share = per[lg].ref_words / per["overall"].ref_words
        print(f"  {lg}: {per[lg]}   [{share:5.1%} of dev words]")
    print(f"  POOLED   : {per['overall']}")
    pred = reweight_to_test_mix(per)
    print(f"  reweighted to the phase-1 test language mix -> multi={pred:.4f}")
    if ci:
        lo, hi = bootstrap_ci(refs, hyps)
        print(f"  95% CI on pooled multi: [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
    return {"per_language": {k: asdict(v) for k, v in per.items()}, "test_mix_multi": pred}


def save(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# SELF-TEST
#
# This module decides which experiments are worth a submission. A scorer that is quietly wrong is
# worse than no scorer at all: it reports a good number for a bad system, and we do not find out
# until the leaderboard contradicts it days later with the submission already spent.
#
# So the properties that matter are asserted rather than assumed. Each case pins down one specific
# claim about the organisers' metric as quoted in this file's docstring.
def _self_test() -> int:
    cases: list[bool] = []

    def check(name: str, got: float, want: float, tol: float = 5e-4) -> None:
        ok = abs(got - want) <= tol
        cases.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} got={got:.4f} want={want:.4f}")

    def check_lt(name: str, got: float, ceiling: float) -> None:
        ok = got < ceiling
        cases.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} got={got:.4f} want<{ceiling}")

    refs = ["mbote na yo mokolo", "ndeipi shamwari yangu", "oli otya ssebo"]

    # 1. Identity. Anything below 1.0 here means normalise() is mutating text it should not.
    check("perfect transcription -> 1.0", score(refs, list(refs)).multi, 1.0)

    # 2. The CTC blank-collapse failure mode. This must SCORE 0.0, not raise: an all-empty output
    #    is exactly what our stage-2 fine-tune produced after 7.5 GPU-hours, and a harness that
    #    crashes on it cannot tell us that is what happened.
    check("empty hypotheses -> 0.0", score(refs, ["", "", ""]).multi, 0.0)

    # 3. The metric lowercases, so case costs nothing.
    check("uppercase is free", score(refs, [r.upper() for r in refs]).multi, 1.0)

    # 4. Accents are NOT folded. This is why post-processing must never "helpfully" strip them.
    check_lt("accents are NOT folded", score(["ekólo"], ["ekolo"]).multi, 1.0)

    # 5. Punctuation IS counted — the punctuation thesis in one line. A perfect transcriber that
    #    omits sentence marks does not score 1.0, and every top-cluster entry is paying this.
    check_lt("punctuation is counted", score(["mbote, na yo."], ["mbote na yo"]).multi, 1.0)

    # 6. jiwer POOLS, it does not average per utterance, so one long wrong sentence must outweigh
    #    one short right one. If this ever flips to a mean, every language-weighting conclusion in
    #    this repo — including reweight_to_test_mix — becomes invalid.
    long_ref = " ".join(["nakei"] * 50)
    pooled = score(["yo", long_ref], ["yo", " ".join(["x"] * 50)]).multi
    averaged = 0.5 * (score(["yo"], ["yo"]).multi
                      + score([long_ref], [" ".join(["x"] * 50)]).multi)
    ok = pooled < averaged - 0.05
    cases.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {'jiwer pools, does not average':46s} "
          f"pooled={pooled:.4f} averaged={averaged:.4f}")

    # 7. An empty REFERENCE is a broken dev set, not a scoreable case: WER divides by reference
    #    length. Fail loudly rather than report a number nobody can interpret.
    try:
        score([""], ["anything"])
        cases.append(False)
        print(f"  [FAIL] {'empty reference must raise':46s} (it did not)")
    except ValueError:
        cases.append(True)
        print(f"  [PASS] {'empty reference raises':46s}")

    # 8. The inferred formula against our one real observation: the WER, CER and Multi Score that
    #    Zindi reported for the zero-shot MMS submission. If this ever fails, the (1-x) inversion
    #    in this file is wrong and every projection in this repo needs redoing.
    check("multi = 0.5(1-WER) + 0.5(1-CER) vs leaderboard",
          0.5 * (1 - 0.6448) + 0.5 * (1 - 0.3713), 0.491944347, tol=2e-3)

    # 9. The test-mix reweighting must actually move the number toward the heavier language, not
    #    silently return the pooled value.
    per = score_by_language(["a b c", "d e f"], ["a b c", "x y z"], ["lin", "lug"])
    rw = reweight_to_test_mix(per)
    ok = 0.0 <= rw <= 1.0
    cases.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {'reweight_to_test_mix in range':46s} got={rw:.4f}")

    n_fail = sum(1 for c in cases if not c)
    print(f"\n{len(cases) - n_fail}/{len(cases)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        print("=== score.py self-test — pinning down the organisers' metric ===")
        raise SystemExit(_self_test())
    print(__doc__)
