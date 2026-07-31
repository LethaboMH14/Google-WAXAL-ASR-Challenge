# HANDOFF — Sbu, read this top to bottom before touching anything

**From:** Lethabo
**Date:** 30 Jul 2026
**Deadline:** competition closes **3 Aug 2026**. That is four days. Nothing here is optional.

> ⚠️ **This repo is PUBLIC.** Never commit `kaggle.json`, a Zindi token, or any API key — the
> `.gitignore` blocks the obvious names but it can't save you from a `git add -f`. Assume
> anything you push here is readable by the other 1,300 entrants.

---

## 0. We are on Lightning AI, not Kaggle — the phone is no longer the blocker

Kaggle will not give you a GPU until the account running the notebooks is phone-verified, and
that verification is blocked on both sides. **So we're not using Kaggle.** We're using
**Lightning AI**: no credit card, no phone verification, 15 free credits a month, and — the
part that actually matters — **persistent storage**.

That last bit removes three whole steps from the old plan. On Kaggle every stage's output had
to be re-uploaded as a Dataset before the next stage could read it. On Lightning, stage N writes
exactly where stage N+1 looks. No uploads, no chance of attaching last week's checkpoint by
mistake, and a training run that dies at hour 6 resumes by **re-running the same command**.

The scripts detect where they're running and resolve paths themselves — the same file works on
Lightning, on Kaggle, and on your laptop. You do not edit paths. If Kaggle ever unblocks, the
Kaggle runbook still works unchanged.

**Pick a T4.** (This used to say "pick an L4, not a T4" — that was written off Lightning's
onboarding shortlist, before we saw the real price list. It was wrong.) Actual rates: T4 **0.19
credits/hr**, L4 **1.58**. The L4 gives about 1.9× the throughput for 8.3× the price, so it is
the worst value on the menu. 15 credits = **79 T4-hours** vs 9.5 L4-hours. Yes, the T4 is Turing
and has no bf16, so training falls back to fp16 — slightly less stable, and a trivial price to pay
at that ratio. See §4.

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
  Test_phase2.csv                   1,500 rows | ID, Target   <- phase 2, the one that pays
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

Two things you need, and one of them is now done:

1. ~~`Test_phase2.csv`~~ **— got it, it's in the repo.** You were right that it wasn't in the
   bulk zip: the Data tab lists five files but the zip's `manifest-*.json` names only four and
   omits this one, so "Download all" silently skips it. Lethabo pulled it separately on 30 Jul.
2. **The 727 MB audio zip — pull it *inside a Kaggle notebook*, not onto your laptop.** Stages
   1 and 3 already `wget` that exact URL into `/kaggle/working`. Kaggle has fast egress and
   20 GB of scratch disk; a home connection does not need to carry this.

### What `Test_phase2.csv` told us — the answer is the bad case

The open question was whether Phase 2 IDs keep the `lin_`/`sna_`/`lug_` prefix that makes
language routing free. Profiled the real file:

```
rows                       1,500          (phase 1 has 4,253)
columns                    ID, Target     Target 100% empty
all match ^ID_[A-Z]{5}$    True           e.g. ID_TBDTM
any lin/sna/lug prefix     False
overlap with phase 1 ids   0
letter freq over 26        0.0360 - 0.0429  (uniform = 0.0385, all 26 used)
```

Five uniformly-random uppercase letters. **No language, and nothing to exploit.** Two
consequences you need to hold onto while you run this:

- **`mms-lid-256` is now load-bearing.** It resolves 0% of Phase 2 from the ID, so LID picks the
  decoder for all 1,500 clips. A LID error isn't a few WER points — it decodes the clip against
  the wrong KenLM and corrupts the entire line.

  **Phase 2 is not lin/sna/lug, and the mix you should expect is nothing like phase 1's.**
  Earlier drafts of this file told you to check the phase-2 mix against ~44/41/15 lin/sna/lug and
  panic if it differed. That guidance was wrong and is withdrawn — it would have you reject a
  *correct* file. Run unconstrained, `mms-lid-256` calls the phase-2 clips **luo, nyn, lug, kin,
  kam, xog** at 0.98–1.00 confidence and returns essentially zero Lingala or Shona; the
  forced-Luganda transcripts the old router produced carry Runyankole grammar and `hu-` verb
  prefixes where Luganda takes `ku-`. Evidence: `local/diagnose_lid_unconstrained.py`, commit
  `e9b3885`. Stage 1 therefore routes **open-set**, over every LID language `mms-1b-all` has an
  adapter for. `WAXAL_CLOSED_SET=1` restores the old three-class behaviour for an A/B.

  So: a phase-2 mix dominated by `luo`/`nyn`/`lug` is the *expected* result. What you should
  check instead is the run's `LID accuracy (open set): X%` line, or produce it separately with
  `python local/calibrate_lid_openset.py` — that measures whether opening the label space makes
  LID leak true-Luganda clips onto their neighbours. ≥90% and the routing is sound.

  **Phase 1 is unaffected either way** — its ids carry `lin_`/`sna_`/`lug_` prefixes and never
  reach LID at all. Its ~44/41/15 mix *is* still the right check for the phase-1 file.
- **Two submission shapes, zero overlap.** Phase 1 wants 4,253 rows in SampleSubmission order;
  Phase 2 wants 1,500 rows in Test_phase2 order. Stages 1 and 3 predict the *union* and write
  **one file per template** — `..._phase1.csv` and `..._phase2.csv`. Upload the one matching
  whichever phase Zindi has open. Never hand-edit one into the other.

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
2. **Lightning AI:** sign up at `lightning.ai`. Free plan, no credit card. You get 15 credits a
   month and one Studio you can leave running.
3. **Create a Studio and clone the repo into it.** The Studio home
   (`/teamspace/studios/this_studio`) is persistent — everything you install or download
   survives a session restart.

   ```bash
   git clone https://github.com/LethaboMH14/Google-WAXAL-ASR-Challenge.git
   cd Google-WAXAL-ASR-Challenge
   bash scripts/setup_lightning.sh
   ```

   The setup script installs the Python stack and the KenLM build dependencies, then prints
   your GPU, whether bf16 is available, and the library versions. **Run it on a CPU Studio** —
   there is no reason to pay GPU rates for a pip install.
4. **There is no data upload step.** The four Zindi CSVs are committed to this repo, so cloning
   it *is* the data setup. The scripts find them at `data/zindi/` automatically.
5. Every stage writes to `/teamspace/studios/this_studio/waxal-work/`. That directory is the
   handoff between stages and it persists. Don't delete it between runs.

### The credit budget — read this before you start a GPU

You have **15 credits/month = 79 hours on a T4** (0.19 credits/hr). The plan needs about 21:

| Stage | Machine | Time | Credits |
|---|---|---|---|
| 0 — LM corpus | **CPU** | ~10-20 min | ~0 |
| 1 — MMS baseline | T4 | ~1 h | ~0.2 |
| 2 — fine-tune | T4 | ~16 h | ~3.0 |
| 3 — KenLM decode | T4 | ~4 h | ~0.8 |
| **Total** | | **~21 h** | **~4.0** |

**Credits are not the constraint. The calendar is.** ~4 of 15 leaves room for three or four
complete re-runs; nothing buys back days before the 3 Aug close. Lightning still bills wall-clock
rather than utilisation, so stop a Studio when a stage finishes — but an idle hour now costs 0.19
credits, not an existential slice of the budget. Optimise the schedule, not the spend.

Two operational facts that will bite you on stage 2:

- **Free Studios stop every 4 hours** and need a manual restart. Stage 2 is ~16 h, so expect
  three or four interruptions. This is normal. Re-run the same command and it resumes from the
  last checkpoint on persistent storage.
- **Auto-sleep after 10 minutes of inactivity.** Fine while a stage is running; it's the gap
  between stages that catches people.

If the calendar tightens, the escape hatch is an **H100 at 4.50 credits/hr with no queue** —
stage 2 in ~2 h for ~9 credits. Most of the month's allowance in one afternoon, and the right
call if it's the difference between submitting a fine-tuned model and not.

---

## 5. The five runs, in order

Run each with plain `python kaggle/<script>.py` from the repo root. Despite the folder name,
these are not Kaggle-only — they detect the environment and resolve their own paths. Each one
prints `env=lightning work=... zindi=...` as its first line; **if that first line says
`env=local` you are not where you think you are.**

**Don't waste credits on stage 0 — it is CPU-only.** Switch the Studio to CPU for it.

> **"Skip stage 2 to save credits" was removed.** It made sense against a 9.5-hour L4 budget.
> On a T4 stage 2 costs ~3 credits of 15 — run it. The skip-stage-2 path (stage 1 + stage 3
> against the baseline model) still exists as a *schedule* fallback if you run out of days, and
> it is a one-flag change if you need it; it is no longer a budget decision.

### Stage 0 — `kaggle/00_build_lm_corpus.py` · **CPU session** · 0 GPU-hours

Builds monolingual text corpora for the language models. Pulls Wikipedia, MasakhaNEWS and
FLEURS transcripts for each of the three languages, plus the in-domain WAXAL text repeated 10×.

- Machine: **CPU Studio.** Do not start a GPU for this.
- Output: `waxal-work/lm_corpus/{lin,sna,lug}.txt` + `lm_sources.json`. Stage 3 reads it from
  there directly — nothing to upload.

**This is the single highest-value stage and its failure mode is silent.** A thin corpus
doesn't crash, it just quietly gives back the biggest win in the plan. The script prints each
corpus as a % of the published target with a verdict — **OK / THIN / TOO SMALL**. Screenshot
that output and send it to me. If Luganda says TOO SMALL, tell me before you go further.

`lm_sources.json` is also our rules-required disclosure of external data. Don't delete it.

### Stage 1 — `kaggle/01_baseline_submission.py` · **T4** · ~1 h · ~0.2 credits

Zero-shot `facebook/mms-1b-all` with its per-language adapters. No training. The point is to
get a real score on the board today and to prove the whole pipeline works — audio resolution →
language routing → decode → submission format — **before** we spend 8 GPU-hours training.

- Output: `waxal-work/submission_01_mms_zeroshot_phase1.csv`, `..._phase2.csv`,
  and `lang_map.json` (the language decisions — stage 3 reuses them instead of re-running LID).
- It downloads the 727 MB phase 2 audio zip into `waxal-work/` on first run. That download is
  persistent, so stage 3 will not repeat it.
- Expect roughly mid-pack. Zero-shot MMS measured 44.7 / 36.9 / 32.1 WER on this corpus. If it
  scores near zero, the format is wrong — check §6 before resubmitting.

Download the CSV, validate it (§6), submit it. **That's our first score.**

### Stage 2 — `kaggle/02_train_w2vbert.py` · **T4** · ~16 h · ~3.0 credits

The actual model: `facebook/w2v-bert-2.0` (580M) fine-tuned as one shared multilingual CTC
model over all three languages. **This is 70% of your credit budget — read §4 first.**

- Output: `waxal-work/w2vbert-waxal/`.
- The script sets `GRAD_ACCUM = 8` on a single GPU (vs 4 on two) so the effective batch size is
  identical either way. You get the same model, it just takes more wall-clock per step.
- **If it dies partway, just run the same command again.** It resumes from
  `waxal-work/w2vbert-waxal/` because that is both where it saves and where it looks. There is
  no upload step and nothing to remember. This is the main thing Lightning buys us.
- Start it and **check back**, don't watch it. But do not let the Studio sit idle after it
  finishes — you are billed for wall-clock, not GPU utilisation.

Watch the first 50 steps. **Loss must be a finite number and must come down.** If it prints
`inf` or `nan`, stop the run and message me — that means the CTC label-length filter is wrong
for some batch and burning 8 hours on it is pointless.

### Stage 3 — `kaggle/03_decode_and_submit.py` · **T4** · ~4 h · ~0.8 credits

Builds a 5-gram KenLM per language from the stage-0 corpora, then decodes the fine-tuned model
with `pyctcdecode` beam search + shallow fusion. **This is where the big win is** — published
measurements on these exact languages show KenLM fusion cutting WER by ~59% on Luganda and
Shona versus greedy decoding.

- Inputs: the stage 0 and stage 2 outputs, both already in `waxal-work/`. Nothing to attach.
- It builds KenLM from source into `waxal-work/kenlm` on first run (several minutes of cmake).
  That build is persistent, so a re-run skips it. If the build fails the script **stops** rather
  than quietly falling back to greedy decoding and costing us the whole LM win.
- It sweeps alpha/beta on the validation split and picks the best per language. It compares
  against greedy and **falls back to greedy if the LM loses** — that's deliberate, not a bug.
- If it warns that the best alpha landed at the edge of the grid, tell me; the grid needs widening.
- Output: `submission_03_w2vbert_lm_phase1.csv` and `..._phase2.csv` — the final submissions.
  It also prints the language mix per file. Sanity-check the **phase 1** file against ~44/41/15
  lin/sna/lug. Do **not** apply that check to the phase 2 file — see the phase-2 note in §3;
  a luo/nyn/lug-dominated mix there is correct, not a misfire.
- **Stage 3 does not yet handle the phase-2 languages, and stage 1 does.** The fine-tuned
  w2v-bert has a CTC vocabulary built from lin/sna/lug transcripts and a KenLM per those three
  languages; neither can produce Dholuo or Runyankole. Until the stage-3 split lands (fine-tuned
  model + KenLM for phase 1 and the lug slice of phase 2, MMS adapters for the rest), stage 1's
  open-set output is the better phase-2 file even though stage 3 is the better phase-1 one. Don't
  assume the newest file wins on both.

### Stage 4 — submit

See §6.

---

## 6. Submitting — the part that is easy to get wrong

**Before every single upload**, on your laptop:

```bash
python local/validate_submission.py path/to/submission.csv
```

It works out which phase your file is by matching its IDs against both templates (the two ID
sets are disjoint, so there's no ambiguity), prints which one it picked, then checks columns,
row count and the exact ID set against *that* template and profiles the predictions against the
training distribution. **If it says FAIL, do not upload.** A malformed file still costs you one
of the 5 daily slots and gives a misleading score.

If it says `validating as phase 1` when you meant to upload phase 2, you grabbed the wrong file
out of `/kaggle/working` — that alone is worth the 5 seconds this takes.

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

## 9. Fixed since your first pass

Both of the things you flagged were real and are in. Pull before you run anything else.

- **`UnicodeEncodeError` on Windows.** Windows consoles default to cp1252 and cannot encode
  `ŋ`, `ᵑ` or `’` — all of which are in the real charset, so the script died partway through
  the report it exists to print. All six scripts now call
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` before any output. No-op on
  Kaggle. **You no longer need `PYTHONIOENCODING=utf-8`** — verified by re-running
  `inspect_data.py` with the environment forced to cp1252, which now completes clean.
- **The "do NOT lowercase blindly" warning.** You were right that it was a false alarm, and it
  was worse than confusing — it predated the starter-notebook finding and was reasoning as if
  we had to match the reference's casing. The scorer lowercases both sides, so casing is free.
  That block is now a *consistency check* against the settled policy, quoting the notebook as
  the source of truth, and it only complains if the data stops matching README §7b.

And the thing you couldn't get:

- **`Test_phase2.csv` is in the repo now.** Your read was right — it's on the Data tab but not
  in the bulk zip. It answered the open question the wrong way: no language prefix (§3). Three
  code changes came out of that, all pushed:
  - Stages 1 and 3 now read **both** templates, predict the union of their IDs, and write one
    correctly-shaped file per template. Before this, stage 1 built its ID list from
    `SampleSubmission.csv` alone and would have silently dropped all 1,500 Phase 2 clips.
  - Both stages print the LID language mix per output file, because LID is now deciding 100%
    of the set that pays.
  - `validate_submission.py` picks its template by ID overlap instead of hard-coding
    `SampleSubmission.csv`. It would otherwise have failed a *correct* Phase 2 file with a
    "row count 1,500 != 4,253" error, which is precisely the false alarm that gets a good
    submission binned.

And the thing that unblocks you:

- **We moved off Kaggle to Lightning AI** (§0). Kaggle's phone verification was blocked on both
  sides, so it stopped being worth solving. Lightning needs no card and no phone, and its
  persistent storage deletes the three Dataset-upload steps between stages.
  - All four stages now detect their environment and resolve paths themselves — Lightning,
    Kaggle, or your laptop, same file. The first line each one prints is `env=... work=...`.
  - Stage 3 builds KenLM into the persistent `waxal-work/` instead of the cwd, so the cmake
    build happens once ever rather than once per session, and it now **asserts** the build
    succeeded instead of silently falling through to greedy decoding.
  - `scripts/setup_lightning.sh` + `requirements-gpu.txt` are new — Kaggle preinstalls this
    stack, Lightning doesn't.
  - Credits are comfortable, the calendar is not: **~4 of your 15 credits** for the full plan. §4 has
    the table. The one way to blow it is leaving a GPU Studio idle.

Good catches. Keep doing that.

---

## 10. Quick reference

| | |
|---|---|
| Competition | Google WAXAL ASR Challenge (Zindi) |
| Languages | Lingala `lin`, Shona `sna`, Luganda `lug` |
| Metric | `0.5·norm(WER) + 0.5·norm(CER)` — higher better |
| Public top | 0.725548135 |
| Closes | 3 Aug 2026 |
| Limits | 5 submissions/day, 200 total, team of 4 |
| Compute | Lightning AI free plan — 15 credits/month, no card, no phone |
| GPU | **T4** (0.19 credits/h, 16 GB, fp16). Not L4 — 8.3× the price for 1.9× the speed |
| Working dir | `/teamspace/studios/this_studio/waxal-work/` — persistent, shared by all stages |
| Seed | 1337 everywhere — never change it |
| Total GPU | ~21 h ≈ 4.0 of your 15 credits (T4 @ 0.19/hr) |
| Biggest risk | Running out of **calendar**, not credits. Free Studios also stop every 4 h |
