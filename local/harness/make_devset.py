"""
Freeze a dev set that predicts the leaderboard, and prove its ids resolve to real audio.

WHY NOT JUST USE THE TEST SET
-----------------------------
The phase-1 test set's ground-truth labels are public on the Hub, and the rules are explicit:
"Any Phase 1 submission that uses the publicly available ground-truth labels for the Phase 1 test
set will be treated as a breach of the challenge rules and may lead to disqualification." We do
not touch them — not to submit, and not to tune, because a hyperparameter chosen against those
labels is laundered into the submission just the same.

We do not need them. `data/zindi/Train.csv` carries an `original_split` column, and 4,220 of its
38k rows are marked `validation`. That is the organisers' own held-out split, handed to us
labelled, entirely within the challenge data. It is the correct dev set and it carries no rules
risk whatsoever.

WHY IT IS SAMPLED RATHER THAN USED WHOLE
----------------------------------------
Two reasons, and only the second is about speed.

1. The leaderboard pools errors across the corpus, so each language's influence on the score is
   its share of reference WORDS. The validation split's language mix is not the test set's. A dev
   set that mirrors the test mix reports a number directly comparable to the leaderboard; one that
   does not needs reweighting every time (score.reweight_to_test_mix does that, but matching the
   mix up front means the raw number is already the right one).
2. Decoding 4,220 clips with a 2.4 GB model and a beam search is not a fast feedback loop. ~900
   clips gets the CI tight enough to resolve the differences we care about, in a fraction of the
   time. DEV_N is the knob; raise it when a decision is close.

The sample is seeded and written to disk, so every experiment scores the SAME utterances. A dev
set that is resampled per run turns model noise and sampling noise into one indistinguishable
number, which defeats the point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "data" / "zindi" / "Train.csv"
TEST_CSV = REPO / "data" / "zindi" / "Test.csv"
OUT = REPO / "local" / "harness" / "devset.json"

SEED = 1337
DEV_N = 900          # total clips; split across languages by the test-set proportions
LANGS = ("lin", "sna", "lug")


def load_train() -> pd.DataFrame:
    # Train.csv has embedded quotes that break the C parser on ~30 rows (a stray `"` inside an
    # unquoted field). engine="python" + on_bad_lines="skip" loses those rows; they are ~0.08% of
    # the file and dropping them from a DEV set costs nothing. It would matter for training, which
    # is why the training script must not reuse this loader without thinking about it.
    return pd.read_csv(TRAIN_CSV, engine="python", on_bad_lines="skip")


def test_mix() -> dict[str, float]:
    """Language proportions of the phase-1 test set, from the id prefixes."""
    ids = pd.read_csv(TEST_CSV)["ID"].astype(str)
    counts = ids.str.split("_").str[0].value_counts()
    return {lg: counts.get(lg, 0) / counts.sum() for lg in LANGS}


def build(n: int = DEV_N, seed: int = SEED) -> dict:
    df = load_train()
    val = df[df["original_split"] == "validation"].copy()
    val["transcription"] = val["transcription"].astype(str)
    # An empty reference makes jiwer raise, and a reference of one character makes WER meaningless.
    val = val[val["transcription"].str.strip().str.len() > 1]

    mix = test_mix()
    rows = []
    for lg in LANGS:
        pool = val[val["language"] == lg]
        want = round(n * mix[lg])
        take = min(want, len(pool))
        if take < want:
            print(f"  !! {lg}: wanted {want} but validation only has {len(pool)} — taking all")
        rows.append(pool.sample(n=take, random_state=seed))
    dev = pd.concat(rows).sort_values("id").reset_index(drop=True)

    manifest = {
        "seed": seed,
        "n": int(len(dev)),
        "source": "Train.csv rows where original_split == 'validation'",
        "target_mix": mix,
        "actual_mix": dev["language"].value_counts(normalize=True).round(4).to_dict(),
        "items": [{"id": r.id, "language": r.language, "reference": r.transcription}
                  for r in dev.itertuples()],
    }
    return manifest


def verify_audio(manifest: dict, per_lang: int = 3) -> None:
    """Stream a few clips and confirm the dev ids actually exist in the Hub validation split.

    Worth the minute it costs. The whole harness is built on the assumption that Train.csv's
    `id` joins to the HF dataset's `id` on the validation split; if that assumption is wrong the
    dev set is unusable and it is much better to find out here than inside a GPU kernel.
    """
    from datasets import Audio, load_dataset

    want = {}
    for it in manifest["items"]:
        want.setdefault(it["language"], [])
        if len(want[it["language"]]) < per_lang:
            want[it["language"]].append(it["id"])

    for lg, ids in want.items():
        need = set(ids)
        found = {}
        ds = load_dataset("google/WaxalNLP", f"{lg}_asr", split="validation", streaming=True)
        # decode=False: this check is about whether the IDS JOIN, not about audio content, and
        # decoding needs torchcodec on datasets>=4 (the local env) while the GPU kernels pin <4
        # and decode through soundfile. Keeping the check decode-free means it runs in both.
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, row in enumerate(ds):
            if row.get("id") in need:
                found[row["id"]] = (row["audio"] or {}).get("path", "<no path>")
            if len(found) == len(need) or i > 6000:
                break
        miss = need - set(found)
        print(f"  {lg}: matched {len(found)}/{len(need)} ids"
              f"{'  MISSING ' + str(sorted(miss)) if miss else '   (join is good)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEV_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verify", action="store_true", help="stream a few clips to check the ids join")
    args = ap.parse_args()

    m = build(args.n, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"wrote {OUT}  n={m['n']}")
    print(f"  target mix (phase-1 test): { {k: round(v, 4) for k, v in m['target_mix'].items()} }")
    print(f"  actual mix (dev set)     : {m['actual_mix']}")
    words = {}
    for it in m["items"]:
        words[it["language"]] = words.get(it["language"], 0) + len(it["reference"].split())
    tot = sum(words.values())
    print(f"  word share (what the metric actually weights by): "
          f"{ {k: round(v / tot, 4) for k, v in sorted(words.items())} }")

    if args.verify:
        print("\nverifying ids resolve to audio on the Hub:")
        verify_audio(m)
