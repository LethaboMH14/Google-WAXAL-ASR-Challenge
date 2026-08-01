# Sbu <-> Lethabo log

Running back-and-forth between Sbu (executing on Lightning AI) and Lethabo (design/plan owner).
Newest entries at the bottom. Pull before you read, push after you write. Keep entries short —
this is a status channel, not the design doc (that's README.md/HANDOFF.md).

---

## 2026-07-31 — Sbu: stage 0 complete on Lightning (CPU studio)

All three languages passed, no THIN/TOO SMALL:

| Language | Corpus words | Target | % | Verdict |
|---|---|---|---|---|
| Lingala | 7,826,099 | 9,025,326 | 86.7% | OK |
| Shona | 19,367,839 | 5,366,168 | 360.9% | OK |
| Luganda | 7,482,535 | 9,168,243 | 81.6% | OK |

Shona overshot because `HPLT2.0_cleaned` hit its `max_words` cap at 12,000,087 words for that
language alone, before the other sources were even added — the script's own safeguard, not an
error.

Two minor notes, neither blocking:
- Saw an "unauthenticated requests to HF Hub, set HF_TOKEN" warning while resolving datasets.
  Everything still completed fine — flagging in case rate limits bite on stage 2/3's larger
  streaming pulls.
- `lm_sources.json` (the rules-required external-data disclosure) wrote successfully.

Also: `00_build_lm_corpus.py`'s closing line still prints `Now: Kaggle -> Output -> New Dataset
-> waxal-lm, mounted by stage 3` — leftover from before the Lightning move. Ignored it; output
already lives in persistent `waxal-work/` and stage 3 reads it directly. Same class of issue as
the stale "prefer L4" line flagged earlier in `setup_lightning.sh`. Two Kaggle-era printouts
found stale now — might be worth a quick sweep across all four scripts for others before they
cause confusion mid-run.

Moving to stage 1 (T4 GPU) next.

---

## 2026-07-31 — Sbu: stage 1 complete, need the calibration number from this run

Ran clean on T4. `submission_01_mms_zeroshot_phase1.csv` and `..._phase2.csv` both wrote,
0 blanks on both (4,253 and 1,500 rows).

Language mix (from `lang_map.json`, derived after the run):

- Phase 1 (id-prefix, no LID involved): lin 1866, sna 1750, lug 638 — matches the documented
  split exactly.
- Phase 2 (LID-routed, all 1,500): lug 302, sna 1, lin 0, and ~1,197 spread across other
  languages — luo 718, nyn 285, kin 107, kam 41, xog 17, wol 7, nso 5, nya 4, ibo 3, umb 3,
  yao 2, swh 2, tha 1, zul 1, bem 1.

First reaction was that this was broken — HANDOFF said to check phase 2 against ~44/41/15
lin/sna/lug and this is nowhere near that. Pulled and read before flagging it though, and found
commit `e9b3885` ("open-set LID for phase 2 — it is not lin/sna/lug") already explains it: the
old 3-class-constrained LID couldn't say "this is Luo," so it forced everything toward Luganda
regardless of truth, and the commit has real evidence for that (wrong verb prefixes, Runyankole
grammar in the forced-lug transcripts). This run used the open-set version from that commit, so
the distribution above looks like the intended behavior, not a new bug.

What I don't have: the run's own printed `LID accuracy (open set): X%` calibration line — it
scrolled out of reach in the terminal and I didn't want to spend another GPU-hour re-running
just to resurface one line. Your commit message says to read that number before trusting the
routing. Can you paste it (or re-run the calibration cell) so we know phase 2's routing is
trustworthy before either of us uploads that file? Phase 1's file doesn't depend on LID at all
(language comes straight from the id prefix), so I'll validate and hold that one ready
regardless.

Also: HANDOFF §3's "check against ~44/41/15" guidance is now stale given the open-set fix —
might be worth updating so nobody else reads that number as the target and panics like I almost
did.

Not uploading anything yet. Waiting on the calibration confirmation.

---

## 2026-07-31 — Sbu: phase 1 validated, OK to upload

Ran `local/validate_submission.py` against `submission_01_mms_zeroshot_phase1.csv` on the
Studio. Result:

```
validating as phase 1 (4,253 rows expected)
rows: 4,253 | words per utt: mean 24.8 (ref 26.0) | uppercase ratio: 0.0000 (ref 0.0133)
WARN  commas present; confirm the CSV quoting survived a round-trip
OK — safe to upload.  (review warnings first)
```

No FAILs, one WARN (generic caution on any comma in the output — only 1 comma across all 4,253
rows, and the file it's reading from is the same round-tripped CSV, so the quoting clearly held).
Distribution vs Train.csv reference looks sane. This one's ready whenever you want to submit it
for the first leaderboard score — not blocked on the phase 2 calibration question above.

Not uploading it myself without your go-ahead, since it's the first submission and I'd rather
you make that call. Say the word and I'll have Sbu submit it, or tell me to hold.

---

## 2026-07-31 — Lethabo: upload phase 1. Calibration script attached. Stale docs fixed.

Good run, and thank you for pulling before flagging the phase-2 distribution — that saved a
round-trip and you got to the right answer on your own.

**1. Upload phase 1. That's the go-ahead.** Reasons, so you can sanity-check the call rather than
just take it: phase 1's languages come from the id prefix and never touch LID, so none of the
open-set uncertainty applies to that file; it costs 1 of 5 daily submissions against 200 total;
and an unknown baseline is worth more to us right now than a marginally better one three days
from now. Push it and paste the score here.

**2. Your phase-2 distribution is the strongest evidence we have, and it's yours.** My call in
`e9b3885` was built on 40 sampled clips. You ran all 1,500 and got luo 718 / nyn 285 / lug 302 /
kin 107 / kam 41 / xog 17 — same shape, 37x the sample, and the tail (wol, nso, ibo, umb, yao,
tha, zul) is small enough to look like genuine LID noise rather than a second population. Add
that to the log as the number we cite from here on; drop my 40-clip one.

**3. The "phase 1 validated OK" vs "FAIL 11 NaN" thing is not a conflict — different files.** I
should say this plainly because on the surface it looks like one of us is wrong. I validated a
CSV built *before* `e9b3885`; that commit bundled three changes, not one: open-set routing, the
`BLANK_FILL` write-out, and apostrophe folding (`git log -S BLANK_FILL` and `-S APOSTROPHES` both
return only `e9b3885`). Your run was on `e9b3885`, which is exactly why you saw 0 blanks and no
`['ʼ', '’']` warning where I saw 11 NaN and the apostrophe warning. Your file is the good one.

**4. The calibration number: `python local/calibrate_lid_openset.py` (new, committed).** You were
right not to burn a GPU-hour re-running stage 1 for one print. The script does LID only over ~100
phase-1 clips per language — no MMS decoding, no submission — using the *identical* routing rule
stage 1 uses, and prints open-set accuracy, closed-set accuracy on the same logits, the confusion
matrix, and the confidence LID had in each wrong answer. CPU is fine, ~10 min, so it won't touch
stage 2's GPU. ≥90% and phase 2 routing is sound; below that it prints what a confidence floor
would need to look like. Paste the verdict here.

Note what that number does and doesn't tell us. It measures whether opening the label space makes
LID leak *known-Luganda* clips onto neighbours — a real risk and the one you're asking about. It
cannot tell us whether the phase-2 references are actually transcribed in Dholuo, because we have
no phase-2 labels and never will before the deadline. My position: even if they aren't, those
clips were producing forced-Luganda garbage before, so open-set can't lose us much and might win
a lot. Uncertainty stated rather than hidden.

**5. Stale docs — all fixed, and I swept the rest as you suggested.** Good catch, and the
`~44/41/15` one was the dangerous one: it would have made someone reject a *correct* phase-2 file.
Fixed in `HANDOFF.md` (both places, with the withdrawal called out explicitly so nobody trusts an
old copy), `README.md`, and the stage-3 printout. Also fixed: the two `Kaggle -> Output -> New
Dataset` lines, the `setup_lightning.sh` claim that an L4 "costs about the same" as a T4 (it does
not — 1.58 vs 0.19 credits/hour on a 15-credit monthly grant, and that line could have quietly
eaten your month), and the "Run on Kaggle GPU" headers on stages 1/2/3.

**6. One thing you should know before it bites you: stage 3 cannot decode phase 2 yet.** It still
routes closed-set, and unlike stage 1 it can't simply be switched — what it decodes *with* is a
w2v-bert fine-tuned on lin/sna/lug transcripts plus one KenLM per those three languages, and
neither can emit Dholuo. So **do not assume stage 3's phase-2 file beats stage 1's.** Stage 3 wins
on phase 1; stage 1's open-set output is currently the better phase-2 file. I've put that warning
in the script and in HANDOFF §3 so it can't be missed at 2am. The fix is a stage-3 split
(fine-tuned + KenLM for phase 1 and the lug slice of phase 2, MMS adapters for the rest) — I'm
designing it now, and if you want to take it instead, say so and it's yours.

Open question I don't have an answer to yet, if you want to dig: do HPLT or GlotCC carry enough
`luo` and `nyn` text to build KenLMs for them? If yes, the non-target clips get shallow fusion
too and the phase-2 file improves a lot. If no, they get greedy MMS and that's the ceiling.

---

## 1 Aug — Lethabo: everything is on Kaggle now, and a real bug in the LM corpus

**Lightning is out.** Not a preference — the paid GPUs are a rules problem ("no paid services or
free trials that require a credit card") and the free tier can't carry stage 2. Everything runs on
Kaggle from here. The cost of that call: the phase-1 and phase-2 CSVs you generated live on that
studio, so as of this morning **we have no submission file at all**. That's the thing I'm fixing
first, ahead of anything clever.

**1. Kaggle's real constraint is 2 concurrent batch sessions, not the GPU count.** I found this by
being refused: `Maximum batch GPU session count of 2 reached`. It counts CPU kernels too — our
CPU LM kernel was holding the slot. So the 30 GPU-hours/week is the budget, but *two jobs at a
time* is the schedule, and that's what actually decides the ordering. Worth knowing before you
queue anything.

**2. Stage 1 now has a kernel (`kaggle/kernels/baseline/`) and is queued behind the LM kernel.**
It's the zero-shot MMS pass — no training, ~1.5 h, writes both phase CSVs — so it's what ends the
no-submission state. Two things the wrapper adds around your script: it runs
`local/validate_submission.py` on both CSVs *inside* the kernel, so a malformed file is caught
before download rather than after it has burned one of five daily slots; and it deletes
`phase2_audio.zip` from `/kaggle/working` afterwards, since everything there becomes the kernel's
output and gets counted against the 20 GB cap.

**3. Found a genuine bug in stage 0, and it's on the lever we care most about.** Train.csv is
17,063 ASCII `'` and **zero** curly apostrophes — I counted — so `'` is the only apostrophe in the
stage 2 CTC vocab. Scraped web text is the opposite: U+2019 everywhere. `normalise()` kept both
characters and folded neither, while `CHARSET` comes from that Train-derived vocab. So we paid
twice, silently: `acceptable()` drops a line once >2% of its characters are outside CHARSET, and
one curly apostrophe in a short sentence is already ~1.5–3%, so whole sentences were being thrown
out of the corpus over punctuation; and the longer lines that passed kept a character the acoustic
model cannot emit, so pyctcdecode could never match those words. It bites hardest exactly where we
need the LM most — Luganda's `ng'` digraph puts an apostrophe inside very common words.

Fixed in `7b9cd95`. **Consequence: the LM kernel running right now was launched before the fix, so
its corpus is the degraded one and needs a rebuild.** I'm letting it finish rather than killing it
— it also produces the open-set LID number you asked for, which this bug doesn't touch — then
rebuilding once stage 1 has its slot. If you get to it first, it's just a re-push of
`kaggle/kernels/lm/`.

**4. Stage 3 was reading its inputs from a mount name that never exists.** On Kaggle a kernel's
output mounts at `/kaggle/input/<kernel-slug>`, so the checkpoint arrives as `waxal-stage2-train`
and stage 1's lang_map as `waxal-baseline` — stage 3 asked for both under `waxal-ckpt`. The
checkpoint load would have failed outright and the lang_map would have been silently skipped,
re-LID'ing every clip. `ART()` now resolves by *content* (find the mount that actually contains
`w2vbert-waxal` / `lm_corpus` / `lang_map.json`) with a `WAXAL_<NAME>` override, so renaming a
kernel doesn't require editing the script. Also added a preflight: the model doesn't load until
after ~5 GB of phase-2 audio has downloaded, so a missing checkpoint now fails in the first second
and prints the mounts it can see.

**5. Your open question is still open** (do HPLT/GlotCC carry enough `luo`/`nyn` text for KenLMs).
The apostrophe fix makes it slightly more likely to be worth it, since the same fold applies to
whatever those corpora give us. Still yours if you want it.

## 1 Aug — Lethabo: stage 2 collapsed to blank. w2v-bert is dropped; MMS is the plan.

**Read this before touching `kaggle/02_train_w2vbert.py`. It does not work and I am not fixing it before close.**

Leg 1 of the w2v-bert fine-tune ran its full 7.5 h on a T4 and produced a model that transcribes
nothing. The wall-clock stop behaved perfectly — stopped and saved at step 998 — which is what
made this easy to misread as a success. It is not one:

| step | eval_loss | eval_wer | eval_cer | eval_score |
|------|-----------|----------|----------|------------|
| 250  | 5.27      | 1.000    | 1.000    | 0          |
| 500  | 2.982     | 1.000    | 1.000    | 0          |
| 750  | 2.967     | 1.000    | 1.000    | 0          |
| 998  | 2.961     | 1.000    | 1.000    | 0          |

WER and CER of *exactly* 1.000 means every hypothesis decoded to the empty string. Train loss
went 154 -> 24.0 by step ~400 and then moved by less than 1% over the next 600 steps
(24.17, 24.08, 24.06, 23.99, 24.13, 24.07, 23.93, 23.92, 24.02, 23.88) while grad_norm fell
167 -> 2. Divide the logged train loss by GRAD_ACCUM=8 and you get ~3.0, which matches eval_loss
2.96 and sits just under ln(46)=3.83 for our 46-token vocab. That is CTC blank collapse: the
model found the all-blank solution, the gradient vanished, and it stayed there.

**Two traps in the artefacts, both of which cost me time:**

1. **The saved model is step 250, not step 998.** `metric_for_best_model="score"` +
   `greater_is_better=True` + every eval scoring 0 means checkpoint-250 was never beaten (a tie
   does not displace the incumbent). `save_total_limit=1` then rotated 500/750/998 away, and
   `load_best_model_at_end=True` reloaded checkpoint-250 before the final `save_model`. The
   only checkpoint dir in the output is `checkpoint-250`. The step-998 weights are gone.
2. **Do not push "leg 2".** The LR-annealing arithmetic in `kaggle/kernels/stage2/waxal_stage2.py`
   is correct and completely beside the point — a well-annealed schedule on a collapsed model
   still emits blanks. Resuming would have burned another 7.5 h to extend a flat line.

**Why it collapsed (my read, stated as a hypothesis, not a finding).** The CTC head is randomly
initialised (`ignore_mismatched_sizes=True`, vocab 46) and WAXAL utterances are long — 176 chars
/ 26 words on average, up to 20 s. Blank is a deep local minimum for CTC and it gets deeper the
longer the target sequence, so a fresh head on long audio is close to the worst case for
bootstrapping an alignment. I did NOT prove this. I checked and cleared the cheap suspects: the
frame-rate maths is right (add_adapter=False -> 50 Hz -> SAMPLES_PER_FRAME=320), the `keep()`
length filter enforces 1.5x frame headroom, the vocab is sane, gradient clipping is on at the
default 1.0, and HF computes the CTC log_softmax in fp32 so this is not fp16 underflow.

**The decision: stop fine-tuning w2v-bert, use MMS.** `facebook/mms-1b-all` already ships trained
CTC adapters for lin/sna/lug. Its heads are pretrained, not random, so the whole failure mode
above is structurally impossible there. Two days to close is not the time to debug a 581M-param
bootstrap. Concretely:

- **Stage 1 (MMS zero-shot) is unaffected and still running.** It is our first real score and it
  never depended on stage 2.
- **Stage 3 gets retargeted onto MMS.** KenLM shallow fusion is the dominant lever in the paper
  (lug 39.75 -> 16.30 WER, sna 22.56 -> 9.28) and it never needed the fine-tune — it needs
  logits, and MMS produces good ones today. Note stage 1 argmaxes its logits inline and does not
  persist them, so fusion is a fresh decode pass, not a post-process on stage 1's CSV.
- **The one real refactor:** stage 3 builds `labels` once from `processor.tokenizer.get_vocab()`.
  MMS swaps vocabulary per language on `set_target_lang()`, so that has to move inside the
  per-language loop, and MMS takes `input_values` (raw waveform) where w2v-bert takes
  `input_features`.

**Sbu — the one thing I want your eyes on:** if we get spare GPU quota, is fine-tuning the MMS
*adapters* (~2M params/lang, pretrained heads) worth it over just decoding zero-shot MMS with a
good KenLM? My instinct is the KenLM is worth more per GPU-hour and carries far less risk, but
that is an instinct, not a measurement.

`kaggle/02_train_w2vbert.py` stays in the repo unchanged — the rules can ask for our code and the
failed attempt is part of the honest record.

---

## 2026-08-01 — Sbu: checked Zindi directly, two things

Committed the Lightning phase1 CSV to `submissions/` for the record (auth fight, not worth
retelling — it's pushed, done). Then checked Zindi's Submissions tab before uploading it and
found you'd already gotten us on the board via Kaggle — did not duplicate-submit.

**1. We have a real score.** `GNXR4Rkc`, ~5h ago: **public score 0.491944347** (CER 0.255986,
WER 0.760125). Checks out against the formula: `1 - 0.5*(0.760125+0.255986) = 0.4919`. First
number on the board.

**2. `M7Ck5P1p` errored** (~5h ago, same batch): *"Wer error: Missing entries for IDs ID_TBDTM,
ID_JZFXM, ID_PKIDG, ID_NORBJ, ID_DNWKH, and more."* Those are phase-2-shaped IDs — looks like an
incomplete file (rows dropped, not a clean phase-1-only submission) rather than a routing
problem. Not investigating further without your read on which run produced it.

**3. Possible date discrepancy — flagging, not assuming.** Zindi's own competition page shows
**Close: 09 Aug 26 (9 days left)**, not the 03 Aug close date HANDOFF has stated throughout. Could
be the page updated after HANDOFF was written, could be a display quirk, could be real. Worth a
5-second look since it changes how much runway we're actually planning against.

---

## 2026-08-01 — Lethabo: answering Sbu's three, and the harness was lying

**Your 3, close date — you're right, and it's 09 Aug.** Confirmed off the live rules page, not
inferred: *Close 09 Aug 26, reveal 10 Aug, "8 days left"*. HANDOFF's 03 Aug is wrong; fix it
wherever it's stated. Same read also pinned down two things worth having in one place:

- **5 submissions per day, 200 overall.** Public LB is ~30% of test, private 70%.
- **2 submissions must be selected before close** for the private leaderboard.
- Metric verbatim: *"the weighted mean of the two evaluation metrics. WER 0.5, CER 0.5"* — which
  is exactly what `local/harness/score.py` implements.
- Top 10 at close get emailed for code, 48h to respond. Winners announced at Deep Learning Indaba
  2–7 Aug.

**Your 2, `M7Ck5P1p`.** That was a phase-2 file, not a broken phase-1 one — `ID_TBDTM` and the
rest are `ID_` + 5 random uppercase letters, the phase-2 id shape. So the grader had phase-2 ids
in its key and our file didn't cover all of them. Worth knowing *why* before we send another
phase-2 CSV: `03_decode_and_submit.py` writes one row per template row, so it cannot drop ids —
unless the template it read was itself short. Two candidates: the phase-2 audio zip didn't yield
all 1,500 clips, or `Test_phase2.csv` wasn't present and it fell back to something smaller. I've
now made the zip reader count and print what it fails to decode instead of silently skipping, so
the next run will say. **Don't upload another phase-2 file until that line reads 1,500.**

**Your 1 — thanks for checking Submissions before uploading.** That's the second time this week
the right move was "look first". Noted.

**And the thing that actually mattered today: our dev harness was lying by +0.2473.**

It predicted 0.7392 for the config that scored 0.4919. Not domain shift — leakage. `corpus_lines()`
and `indomain()` both built the KenLM from *all* of `Train.csv`, and `Train.csv` contains the
`original_split == "validation"` rows that **are** our 900-clip dev set. The LM had memorised the
dev references verbatim and shallow fusion decoded those same clips against it.
`00_build_lm_corpus.py` was worse, because `INDOMAIN_REPEAT` upweights that text.

Held out by id and by normalised content, in both files, unconditionally. **Every alpha/beta pair
we have was tuned on leaked data and is void.**

**And I was wrong about punctuation.** I wrote in `docs/MODEL-CANDIDATES.md` that "the way past
them is punctuation, not a bigger acoustic model." The LM-free bakeoff (10 candidates, frozen dev
set) says the opposite — acoustic is the lever, punctuation is a garnish:

| lang | control (mms-300m) | winner | delta |
|---|---|---|---|
| lin | 0.6893 | `douyeszn/w2vbert-lin-waxal-aug-ft` **0.7788** | **+0.0895** |
| sna | 0.7815 | `Mubarak127/waxal-whisper-large-v3-sna_asr` **0.8034** | +0.0219 |
| lug | 0.8163 | nothing beat the control | — |

Trailing '.' is worth +0.004 lin, +0.012 lug, and **−0.018 sna** (Whisper already punctuates; the
append makes `..`). So it's per-language now, `WAXAL_PLUS_PERIOD="lin,lug"`.

Picking checkpoints by reading `vocab.json` also failed twice: `keystats` has full punctuation and
still lost lin by 0.08, and the punctuated `douyeszn/w2vbert-lug-waxal-aug` scored **exactly
0.0000** — CTC blank collapse, WER and CER both 1.000. Run them, don't read them.

Word-weighted estimate for the combined lineup is **0.7984**; kernel `waxal-lineup` replaces that
estimate with a pooled three-language dev measurement and then writes the CSVs from the identical
config. **Nitpick request, per CLAUDE.md §7:** the calibration argument I'm leaning on is that our
harness puts the organisers' mms-300m set at 0.7453 while the LB top cluster is 0.7206–0.7257,
implying a ~+0.02 bias. That assumes the leaders run those checkpoints — inferred from download
counts, not observed. If you can break that assumption, break it now rather than after we've spent
submissions on it.

**One thing back at you:** `submissions/submission_01_mms_zeroshot_phase1.csv` is now committed to
a **public** repo. Not a rules problem — they're our own predictions — but it does hand our
baseline to anyone reading, and `.gitignore` un-ignores `submissions/`, so a `git add -A` will
publish every future CSV in there too. Your call whether to keep it; flagging so it's a decision
rather than an accident. I've been staging explicit paths for exactly this reason.

---

## 2026-08-01 — Lethabo: `M7Ck5P1p` resolves the other way, and it matters

Checked our phase-2 file against the template rather than guessing:

```
Test_phase2.csv                            1500 rows
submission_01_mms_zeroshot_phase2.csv      1500 rows
missing: 0    extra: 0
ID_TBDTM, ID_JZFXM, ID_PKIDG, ID_NORBJ, ID_DNWKH — all present
```

So the file the grader complained about was **not** our phase-2 CSV. Every id it named as missing
is in it. Which forces the opposite reading of the pair, and the logic only closes one way:

- Phase-1 and phase-2 id sets are **disjoint** (verified: overlap = 0).
- `M7Ck5P1p` errored with *"Missing entries for IDs ID_TBDTM, …"* — **phase-2** ids.
- A file missing every phase-2 id **is** the phase-1 file.
- `GNXR4Rkc` scored 0.491944. A phase-1 file cannot score against a phase-2 key — it would have
  thrown the same error.

**Therefore the grader's key is phase 2. `GNXR4Rkc` was the phase-2 CSV, and our 0.4919 is a
phase-2 score.** `M7Ck5P1p` was the phase-1 CSV and was correctly rejected.

Three consequences, and the first one is the one that costs us if we get it wrong:

1. **Upload the phase-2 CSV.** A phase-1 upload burns one of five daily submissions on a
   guaranteed error. That is what `M7Ck5P1p` cost.
2. **0.4919 already includes our LID routing.** Phase-2 ids carry no language metadata, so that
   score is the acoustic model *and* the router together. Any router improvement shows up in the
   leaderboard number directly — which is why `waxal-lid-probe` is worth its GPU time and not a
   side quest.
3. **The dev harness is measuring the right thing** — it routes with the same LID model rather
   than reading language off an id prefix.

Caveat, stated because I can't close it: this is inference from the error text, not from the
Submissions tab. I'm signed out of Zindi in this browser and Submissions needs auth. If you're
signed in, one look confirms or kills it — check whether `GNXR4Rkc` shows 1,500 rows or 4,253.
Until then I'd treat it as strong but not certain, and still upload phase 2, because phase 2 is
the correct file under **both** readings: it's complete, and it's the split that sets the final
private ranking (70% of test).
