# HANDOFF — Sbu, read this top to bottom before touching anything

**From:** Lethabo
**Date:** 30 Jul 2026
**Deadline:** competition closes **3 Aug 2026**. That is four days. Nothing here is optional.

> ⚠️ **This repo is PUBLIC.** Never commit `kaggle.json`, a Zindi token, or any API key — the
> `.gitignore` blocks the obvious names but it can't save you from a `git add -f`. Assume
> anything you push here is readable by the other 1,300 entrants.

---

## 0. Why you and not me

Kaggle will not give you a GPU until the account is **verified with a phone number**. I don't
have a usable phone for that right now, you do. That's the whole reason this is landing on you.

Everything else is already done: the data is downloaded and profiled, the plan is decided and
written up, and all six scripts are written, corrected against the real data, and compiling.
**You are not designing anything. You are executing five runs and uploading CSVs.** If you find
yourself rewriting model code, stop and message me — something has gone wrong upstream.

Read `README.md` after this file if you want the reasoning. This file is the instructions.

---

## 1. STOP — three rules that get us disqualified, not just marked down

These are quoted from the Zindi rules page. Break any one and the whole thing is void.

1. **We are already a registered team on Zindi — keep it that way.**
   > "Multiple accounts, or sharing of code and information across accounts not in teams, is
   > not allowed and will lead to disqualification."

   Confirmed done. That is what makes it legal for you to have this code. Max team size is 4,
   so if anyone else joins, add them on the **Team** tab *before* sharing anything with them.
   Submissions go under the team, so don't create a second account for any reason.

2. **Never join the HuggingFace `test` labels onto the submission.**
   > "Any Phase 1 submission that uses the publicly available ground-truth labels for the
   > Phase 1 test set will be treated as a breach of the challenge rules."

   The HF `test` split of `google/WaxalNLP` **does** contain the answers. You could join on
   `id` and score ~1.0 and be disqualified within a day. Scripts `01` and `03` load audio from
   that split and drop the `transcription`/`text` columns on sight — the line is commented
   `# === RULES GUARD ===`. **Do not remove it, do not "just check" what's in there.**

3. **No paid or closed services.** Open-source tools and openly-available pretrained models
   only. No Azure Speech, no OpenAI API, no AutoML. Everything in the plan is HuggingFace +
   KenLM, all free and open.

Also: **always keep `SEED = 1337`.** Rerunning must reproduce the leaderboard position — that
is a rule, and if we finish top 10 they email us and we have **48 hours** to hand over code
that reproduces.

---

## 2. What the task actually is

Build an ASR system for three low-resource Bantu languages — **Lingala (`lin`),
Shona (`sna`), Luganda (`lug`)** — using the Google WAXAL corpus.

- **Metric:** `0.5 × norm(WER) + 0.5 × norm(CER)`. **Higher is better.**
- **Public leaderboard top right now:** `0.725548135`. That is roughly the level of the
  corpus authors' own published paper, so beating it means beating published SOTA. It is
  doable — the plan in `README.md §2` says how — but nobody is getting 0.95 here. If a number
  looks too good, we broke rule 2 above.
- **Submissions:** 5 per day, 200 total. Don't burn them on untested files.
- **Phase 2 is what pays.** The final ranking uses a brand-new unseen test set with **no
  metadata at all** — no language column, no speaker, no gender. That is why we train one
  shared multilingual model plus audio language-ID, and not three per-language models.

---

## 3. The data — it's in this repo

```
data/zindi/
  Train.csv                        38,199 rows | id, transcription, language, original_split
  Test.csv                          4,253 rows | ID   <- that is the ONLY column
  SampleSubmission.csv              4,253 rows | ID, Target
  Waxal_Challenge_Starter_Code.ipynb           organisers' reference implementation
  Test_phase2.csv                  MISSING — see below, get this on day one
```

If `data/zindi/` is empty when you clone, download the files yourself from the competition
page → **Data** tab (you need to be logged in and have accepted the terms). Put them in
`data/zindi/` with exactly those filenames.

### PHASE 2 IS ALREADY OPEN — this is the most urgent item in this document

Checked 30 Jul: `https://storage.googleapis.com/waxalphase2/audio.zip` returns **HTTP 200,
762,423,240 bytes (727 MB), Last-Modified 27 Jul 2026**. The organisers said Phase 2 audio
drops "approximately one week before the challenge closes" and it did, three days ago.

**Phase 2 is what determines the final rankings and the prize money.** The Phase 1 leaderboard
is explicitly developmental:

> "The Phase 1 leaderboard is designed to support model development, collaboration and
> experimentation. Final rankings and prize winners will be determined based on performance on
> the Phase 2 evaluation dataset."

Two things to get, and neither is optional:

1. **`Test_phase2.csv` (14.7 KB) — Zindi Data tab, download it by itself.** The Data tab lists
   five files but the bulk zip's `manifest-*.json` names only four and omits this one, so
   "Download all" silently skips it. It's the ID list: which clips to predict, and the row
   order the submission must be in. Put it in `data/zindi/` and tell me it's there.
2. **The 727 MB audio zip — pull it *inside a Kaggle notebook*, not onto your laptop.** Stages
   1 and 3 already `wget` that exact URL into `/kaggle/working`. Kaggle has fast egress and
   20 GB of scratch disk; a home connection does not need to carry this.

**Tell me two things the moment you have `Test_phase2.csv`:** how many rows, and whether the
IDs still start with `lin_` / `sna_` / `lug_`. If the prefix is gone — which is what "metadata
will not be provided" implies — then `mms-lid-256` language ID is doing real work on the set
that pays, and it is worth spending a submission slot checking it before the deadline. If the
prefix survived, routing stays free and we have one less thing to worry about.

**Three things about this data you must not rediscover the hard way:**

- **Always `pd.read_csv(..., escapechar="\\")`.** Zindi backslash-escapes quotes inside
  quoted fields. Without it pandas dies with `Expected 4 fields in line 9570, saw 5` on 23 of
  the 38,199 rows. Every read in this repo already does it. If you write new code, do it too.
- **The audio is not in the CSVs.** `Test.csv` is a bare list of IDs. Audio comes from the
  HuggingFace dataset `google/WaxalNLP`, configs `lin_asr` / `sna_asr` / `lug_asr`. The
  scripts stream it — you need **Internet ON** in the Kaggle notebook settings.
- **IDs carry the language.** `lug_96114` → `lug`. All 4,253 submission IDs resolve
  (lin 1,866 / sna 1,749 / lug 638), so Phase 1 needs no language-ID model. The LID model is
  in there for Phase 2, which won't have the prefix.

---

## 4. Setup — do this once, in this order

1. **Zindi:** create an account, join the challenge, accept the data terms, **and accept my
   team invite** (rule 1 above).
2. **Kaggle:** create an account → Settings → **Phone Verification**. Without it there is no
   GPU and none of this works. Then confirm you can see "GPU T4 x2" in a notebook's
   Accelerator dropdown.
3. **Upload the data as a Kaggle Dataset.** Kaggle → Datasets → New Dataset → drag in the
   four files from `data/zindi/`. Name it **exactly** `waxal-zindi` — the scripts hardcode
   `/kaggle/input/waxal-zindi`. Set it **Private**.
4. For each stage below: new Kaggle Notebook → paste the script in → Settings: **Accelerator**
   as stated, **Internet ON**, **Persistence: Variables and Files** → attach the input
   datasets it needs → **Save Version → Save & Run All (Commit)**.

   Run it as a *committed* version, not interactively. Interactive sessions die when your
   laptop sleeps; committed runs finish on Kaggle's side.

---

## 5. The five runs, in order

Budget is ~30 GPU-hours per week on the free tier. The plan needs ~11.5, so there is room for
one failed run. **Don't waste GPU on stage 0 — it is CPU-only.**

### Stage 0 — `kaggle/00_build_lm_corpus.py` · **CPU session** · 0 GPU-hours

Builds monolingual text corpora for the language models. Pulls Wikipedia, MasakhaNEWS and
FLEURS transcripts for each of the three languages, plus the in-domain WAXAL text repeated 10×.

- Attach: `waxal-zindi`. Accelerator: **None**. Internet: **ON**.
- Output: `/kaggle/working/lm_corpus/{lin,sna,lug}.txt` + `lm_sources.json`.
- **When it finishes: Save Version output → New Dataset named `waxal-lm`.**

**This is the single highest-value stage and its failure mode is silent.** A thin corpus
doesn't crash, it just quietly gives back the biggest win in the plan. The script prints each
corpus as a % of the published target with a verdict — **OK / THIN / TOO SMALL**. Screenshot
that output and send it to me. If Luganda says TOO SMALL, tell me before you go further.

`lm_sources.json` is also our rules-required disclosure of external data. Don't delete it.

### Stage 1 — `kaggle/01_baseline_submission.py` · **GPU T4 x2** · ~1.5 h

Zero-shot `facebook/mms-1b-all` with its per-language adapters. No training. The point is to
get a real score on the board today and to prove the whole pipeline works — audio resolution →
language routing → decode → submission format — **before** we spend 8 GPU-hours training.

- Attach: `waxal-zindi`. Internet: **ON**.
- Output: `/kaggle/working/submission_01_mms_zeroshot.csv` and `lang_map.json`.
- Expect roughly mid-pack. Zero-shot MMS measured 44.7 / 36.9 / 32.1 WER on this corpus. If it
  scores near zero, the format is wrong — check §6 before resubmitting.

Download the CSV, validate it (§6), submit it. **That's our first score.**

### Stage 2 — `kaggle/02_train_w2vbert.py` · **GPU T4 x2** · ~8 h

The actual model: `facebook/w2v-bert-2.0` (580M) fine-tuned as one shared multilingual CTC
model over all three languages.

- Attach: `waxal-zindi`. Internet: **ON**. Persistence **ON**.
- Kaggle kills a session at 12 h, so 2,500 steps is sized to fit with margin.
- **When it finishes: Save Version output → New Dataset named `waxal-ckpt`.**
- If it dies partway, upload whatever checkpoint exists as `waxal-ckpt` and re-run — the
  script resumes from `/kaggle/input/waxal-ckpt`.

Watch the first 50 steps. **Loss must be a finite number and must come down.** If it prints
`inf` or `nan`, stop the run and message me — that means the CTC label-length filter is wrong
for some batch and burning 8 hours on it is pointless.

### Stage 3 — `kaggle/03_decode_and_submit.py` · **GPU T4 x2** · ~2 h

Builds a 5-gram KenLM per language from the stage-0 corpora, then decodes the fine-tuned model
with `pyctcdecode` beam search + shallow fusion. **This is where the big win is** — published
measurements on these exact languages show KenLM fusion cutting WER by ~59% on Luganda and
Shona versus greedy decoding.

- Attach: `waxal-zindi` **and** `waxal-ckpt` **and** `waxal-lm`. Internet: **ON**.
- It sweeps alpha/beta on the validation split and picks the best per language. It compares
  against greedy and **falls back to greedy if the LM loses** — that's deliberate, not a bug.
- If it warns that the best alpha landed at the edge of the grid, tell me; the grid needs widening.
- Output: the final submission CSV.

### Stage 4 — submit

See §6.

---

## 6. Submitting — the part that is easy to get wrong

**Before every single upload**, on your laptop:

```bash
python local/validate_submission.py path/to/submission.csv
```

It checks the columns, row count and the exact ID set against `SampleSubmission.csv`, and
profiles the predictions against the training distribution. **If it says FAIL, do not upload.**
A malformed file still costs you one of the 5 daily slots and gives a misleading score.

Then: competition page → **Submit** → drag the CSV → add a comment saying which stage it came
from (you will not remember on Sunday) → Submit.

**Before the competition closes you must SELECT 2 SUBMISSIONS** to be judged on the private
leaderboard. If you select nothing, Zindi uses your best 2 public scores — which is usually
fine but not always, because the public set is only part of the data. Set a reminder.

---

## 7. What to send me after each stage

Short messages, not essays:

- **Stage 0:** the OK / THIN / TOO SMALL verdict lines, and the word count per language.
- **Stage 1:** the leaderboard score, and the "blank=" percentage the script prints.
- **Stage 2:** loss at step 50, and the eval score at each checkpoint (every 500 steps).
- **Stage 3:** the chosen alpha/beta per language, and greedy-vs-LM WER for each.
- **Anything that crashes:** the full traceback, not a description of it.

---

## 8. Nitpick it, don't just run it

Same standard as VUKA. If something in here doesn't actually work on Kaggle's image, or a
library version has moved, or the logic doesn't match what the competition is really scoring —
**say so before you build around it**. Two things I already know are open and would like a
second opinion on:

1. **Punctuation.** The organisers' starter notebook scores by lowercasing both sides and
   leaving punctuation alone, so we keep punctuation. But I can't verify Zindi's *server-side*
   scorer does the same. Cheap test: spend two of the five daily slots on identical predictions
   with and without punctuation and read the delta off the leaderboard. Worth doing on day one.
2. **Whether we are even optimising the right thing.** Phase 2 is open (§3) and it is what
   pays, but every score we can *see* comes from Phase 1. Those are different distributions —
   Phase 2 is new speakers and new recordings by construction. If tuning alpha/beta hard
   against Phase 1 starts looking like overfitting to you, say so; the safer choice is the
   setting that wins on the *validation* split, not the one that wins on the public
   leaderboard. That judgement call is worth more than a few decimal places.

---

## 9. Quick reference

| | |
|---|---|
| Competition | Google WAXAL ASR Challenge (Zindi) |
| Languages | Lingala `lin`, Shona `sna`, Luganda `lug` |
| Metric | `0.5·norm(WER) + 0.5·norm(CER)` — higher better |
| Public top | 0.725548135 |
| Closes | 3 Aug 2026 |
| Limits | 5 submissions/day, 200 total, team of 4 |
| Kaggle datasets | `waxal-zindi` (you upload), `waxal-lm` (stage 0), `waxal-ckpt` (stage 2) |
| Seed | 1337 everywhere — never change it |
| Total GPU | ~11.5 h of the ~30 h/week free allowance |
