# WAXAL ASR — Zindi / Google Research

Target: Google WAXAL ASR Challenge. Lingala (`lin`), Shona (`sna`), Luganda (`lug`).
Metric: `0.5 * norm(WER) + 0.5 * norm(CER)` — higher leaderboard score is better.
Close: **3 Aug 2026**.

---

## 1. What score is actually achievable

The dataset has a published benchmark paper, [WAXAL-NET (arXiv:2606.02375)](https://arxiv.org/html/2606.02375),
run on **A100s** by the people who built the corpus. Their best numbers on these three
languages:

| Language | Train hrs | Best WER | Best CER | Best system |
|---|---|---|---|---|
| Lingala | 71.9 | **42.6%** | **18.9%** | MMS-300M (CTC, greedy) |
| Shona | 79.7 | **25.0%** | **4.3%** | MMS-300M (CTC, greedy) |
| Luganda | 37.3 | **16.9%** | **3.4%** | MMS-300M (CTC, greedy) |

Zero-shot Whisper-large on the same data: **93% / 112% WER**. Whisper is not a shortcut here.

Mean error across the three ≈ 28% WER / 9% CER → `1 - 0.5*(0.28+0.09) ≈ 0.815` if Zindi
normalised naively, but the observed leaderboard top is **0.7255**, which is consistent with
the leaders sitting at roughly **published-SOTA level**. So:

- **Near-perfect is not physically available.** This is spontaneous, image-prompted,
  multi-speaker speech in three low-resource Bantu languages. Nobody gets 5% WER here.
- **Beating 0.7255 means beating the corpus authors' own published result.** That is the
  actual bar. It is reachable, and §2 says exactly how.

Anyone claiming a near-1.0 score on Phase 1 is joining the HF `test` split labels on `id` —
that is an explicit disqualification under the rules. We do not do that. See §6.

---

## 2. Where the wins are (ranked by expected gain per hour spent)

1. **KenLM shallow fusion + beam search — and it is not close.** WAXAL-NET decoded **CTC
   greedy**. [arXiv:2512.10968](https://arxiv.org/html/2512.10968) measured w2v-bert-2.0 with
   and without a 5-gram KenLM on *exactly these three languages*:

   | Language | w2v-bert greedy | w2v-bert + KenLM | Relative |
   |---|---|---|---|
   | Luganda (200h) | 39.75 | **16.30** | **−59%** |
   | Shona (100h) | 22.56 | **9.28** | **−59%** |
   | Lingala (100h) | 24.19 | **22.74** | −6% |

   Those absolute numbers are on CommonVoice/FLEURS read speech and do **not** transfer to
   WAXAL's spontaneous speech — but the mechanism does, and it is far larger than the 15–25%
   rule of thumb. The LM is the deliverable; the acoustic model is the thing that feeds it.

   **The catch, and the reason `kaggle/00_build_lm_corpus.py` exists:** those LMs were built
   on **9.0M (lin) / 9.2M (lug) / 5.4M (sna) words**. Measured, `Train.csv` gives us:

   | | utterances | words | vs published LM corpus |
   |---|---|---|---|
   | Lingala | 16,244 | 445,865 | 4.9% |
   | Shona | 15,836 | 370,661 | 6.9% |
   | Luganda | 6,119 | 177,390 | **1.9%** |

   Five to fifty times short, Luganda worst — and Luganda is where the LM gain was largest.
   A 5-gram that sparse is a lookup table for the training set, and the same paper shows LM
   fusion making things *worse* in exactly that regime (XLS-R+LM regresses on both Lingala
   and Shona). So we pull real monolingual text from open corpora first. Stage 3 still tunes
   against greedy per-language and falls back if the LM loses, so a thin corpus is survivable —
   just not a win.
2. **w2v-bert-2.0 instead of MMS-300M.** 580M params, pretrained on 4.5M hours / 143
   languages. In the benchmark above it beats MMS and XLS-R at **every** data scale on all
   three languages, before the LM is even applied. *Expected: a further 5–10% relative.*
3. **Text normalisation matched to the reference.** WER is unforgiving about casing,
   punctuation and diacritics. Getting the output distribution to match the reference
   distribution exactly is worth several WER points and costs nothing. `local/inspect_data.py`
   measures this against the real `Train.csv` instead of guessing.
4. **One multilingual model, not three.** Required anyway — see §3.
5. **Pseudo-labelling the `unlabeled` split** (109k clips/lang). Real gains, but does not fit
   in 3 days on free compute. Listed for completeness, not scheduled.

---

## 3. The thing most competitors will get wrong

**Phase 2 decides the prizes, and Phase 2 ships no metadata.**

> "Metadata and auxiliary information such as language, speaker identity, gender ... will not
> be provided."

So a per-language model keyed off a `language` column silently breaks on the set that
actually counts. Our design is therefore:

- **one shared multilingual CTC acoustic model** (shared Latin-script vocab across lin/sna/lug), plus
- **explicit language ID** from the audio (`facebook/mms-lid-256`), used only to pick which
  KenLM to decode with, with LM score as the tiebreak.

Phase 1 leaderboard is developmental. We submit to it to validate the pipeline, but Phase 2
is the deliverable.

---

## 4. Compute reality

This laptop is an i5-1235U / 7.7 GB RAM / **Intel UHD integrated graphics — no CUDA**.
It cannot train, or even comfortably infer, a 600M-param acoustic model.

**Azure is not an option for training here, and this was checked rather than assumed.**
The subscription is *Azure for Students*. Measured GPU quota:

| SKU | GPU | Quota (eastus) | Quota (southafricanorth) |
|---|---|---|---|
| `Standard_NC4as_T4_v3` | T4 | **0** | not offered |
| `Standard_NC6s_v3` | V100 | **0** | not offered |
| `Standard_NV6ads_A10_v5` | A10 | **0** | **0** |
| `Standard_NV4as_v4` | partial AMD MI25 | 2 | not offered |

Every CUDA SKU is zero. `NV4as_v4` has quota but is a fractional AMD GPU for virtual desktops —
no CUDA, useless for PyTorch. Raising GPU quota requires upgrading to Pay-As-You-Go (credit
card) and then waiting on an approval that typically takes days; the challenge closes in three.
The $100 credit is not the binding constraint — the quota is.

**Also: do not reach for Azure Speech or Azure OpenAI.** The rules say *"You may only use
open-source languages and tools"*. A closed managed API would be disqualifying regardless of
whether it worked, and it does not meaningfully support lin/sna/lug anyway.

Azure is still useful for the **CPU** side — a `Standard_D4s_v3` (4 vCPU / 16 GB, quota
available) beats this laptop for KenLM building and the alpha/beta sweep, both of which are
CPU-only.

**Training runs on Kaggle** (free: 2×T4 16GB, ~30 GPU-hours/week, session-capped, internet on,
no credit card). Fallback ladder if Kaggle is unavailable:

| Option | GPU | Cost | Notes |
|---|---|---|---|
| **Kaggle** | 2×T4 | free | 30 GPU-h/week. Enough for the whole plan. Default. |
| Lightning AI Studio | T4 | free tier | ~22 GPU-h/month, persistent disk, no CC |
| Colab free | T4 | free | disconnects hard; only for stage 1 and 3 |
| Colab Pro | L4/A100 | ~$10/mo | best value if any spend is possible |
| RunPod / Vast.ai | 4090/A5000 | ~$0.25/h | ~$5 for the full run; needs a card |

Renting compute does not breach the rules — *"no paid services or free trials that require a
credit card"* sits in the **code-reproducibility** section and constrains what the *solution
depends on* (paid APIs, AutoML), not where you rent a GPU. Otherwise no one with a personal
GPU could enter. Post the question on the Zindi discussion board if you want it in writing.

Budget for the ~30 available Kaggle GPU-hours:

| Stage | Script | GPU-h |
|---|---|---|
| LM text corpora | `kaggle/00_build_lm_corpus.py` | **0** (CPU session) |
| Zero-shot baseline → first valid submission | `kaggle/01_baseline_submission.py` | ~1.5 |
| Multilingual w2v-bert-2.0 CTC fine-tune | `kaggle/02_train_w2vbert.py` | ~8 (one session; a 2nd is upside) |
| LM build + beam decode + LID + final submission | `kaggle/03_decode_and_submit.py` | ~2 |
| Slack for one failed run | — | ~10 |

The local machine's job is orchestration, data inspection and submission validation only.

---

## 5. Order of operations

```
0. Download the 5 Zindi files into data/zindi/   (you, manually — they are behind login)
1. python local/inspect_data.py                  (locally — decides normalisation + format)
2. kaggle/00_build_lm_corpus.py                  (Kaggle CPU — no GPU quota burned)
                                                 -> save output as Dataset `waxal-lm`
3. kaggle/01_baseline_submission.py              (Kaggle GPU — get a score on the board today)
4. kaggle/02_train_w2vbert.py                    (Kaggle GPU — the actual model)
                                                 -> save output as Dataset `waxal-ckpt`
5. kaggle/03_decode_and_submit.py                (Kaggle GPU — LM decode, both phases)
6. python local/validate_submission.py <csv>     (locally — before every upload)
```

Stage 0 runs on a **CPU** session, so it costs none of the 30 weekly GPU-hours, and it can run
concurrently with stage 1. Do it early: it is the only stage whose failure mode is silent
(a thin corpus doesn't crash, it just quietly gives back the biggest win in the plan).

Submission mechanics: Zindi competition page → **Submit** → drag the CSV → optional comment
→ Submit. Limit **5/day, 200 total**. Before close, select the **2** submissions to be judged
on the private leaderboard; if you select nothing, your best 2 public scores are used.

---

## 6. Rules compliance (do not skip)

- The HuggingFace `test` split of `google/WaxalNLP` **contains ground-truth transcriptions**,
  and the Phase 1 test set is drawn from it. Joining on `id` would score ~1.0 and is an
  explicit disqualification: *"Any Phase 1 submission that uses the publicly available
  ground-truth labels for the Phase 1 test set will be treated as a breach of the challenge
  rules."* All training code here trains on the `train` split and validates on `validation`.
  `test` is never loaded with labels.
- Top 10 on the private leaderboard get a **code review request with a 48-hour deadline**.
  Every script here sets seeds and pins versions so the result reproduces.
- External public datasets are allowed but **must be disclosed** in the final documentation.
  We use two kinds:
  - **Pretrained checkpoints** — `facebook/w2v-bert-2.0`, `facebook/mms-1b-all`,
    `facebook/mms-lid-256`. All openly available to everyone, which is the rule's test.
  - **Monolingual text for the language models** — Wikipedia (CC-BY-SA-4.0),
    [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) (AFL-3.0), and
    [FLEURS](https://huggingface.co/datasets/google/fleurs) transcripts (CC-BY-4.0), per
    language. `kaggle/00_build_lm_corpus.py` writes **`lm_corpus/lm_sources.json`** with the
    exact repo, config, sentence count, word count and licence for every source actually used —
    that file *is* the disclosure, so it can never drift from what the code did. No audio from
    these datasets is used for training; text only, for the LM.

---

## 7. What the real data changed (measured 30 Jul, not assumed)

The Zindi archive is extracted into `data/zindi/`. Reading it overturned three decisions I had
already written into the scripts. All four findings below are load-bearing.

**a. `Train.csv` breaks pandas without `escapechar`.** Zindi backslash-escapes quotes inside
quoted fields (`\"`), which is not standard CSV. The C parser reads the escaped quote as a
field terminator and dies: `ParserError: Expected 4 fields in line 9570, saw 5`. **23 ragged
rows out of 38,199.** Every `read_csv` in this repo (13 call sites, 6 files) now passes
`escapechar="\\"`. Dropping those rows instead would silently bin 23 transcripts.

**b. Do not strip punctuation.** I had `STRIP_PUNCT = True`. The organisers' own
`Waxal_Challenge_Starter_Code.ipynb` scores with:

```python
refs_lower = [r.lower() for r in references]
preds_lower = [p.lower() for p in predictions]
return {"wer": jiwer.wer(refs_lower, preds_lower), "cer": jiwer.cer(refs_lower, preds_lower)}
```

Lowercase both sides, punctuation **untouched**. Punctuation is in ~9.7% of reference tokens
(67,456 periods, 28,620 commas in 993,916 words), so a model that never emits it eats those as
substitutions. Policy is now `LOWERCASE = True`, `KEEP_PUNCT = {. , ' ’ - ; : ! ?}`, applied
identically in stages 0/1/2/3. Charset is frequency-gated at `MIN_CHAR_COUNT = 25` → **vocab 46**,
with 18 rare accented chars folded to their unaccented base (é→e) rather than mapped to `[UNK]`.

**c. `add_adapter=False`.** I had it on. Measured label lengths on the real transcripts:
mean 176 chars, p50 169, p95 305, max 650. The adapter's stride-2 conv halves the CTC frame
rate from 20 ms to 40 ms, so a 12 s clip yields 300 frames for ~169 chars — **1.8 frames/char**.
CTC needs a blank between every repeated symbol, and Bantu orthography is full of doubled
letters (*ekkubo*, *ssukuma*, *ennyaanya*, *amaato*). At 1.8 frames/char those utterances are
not merely hard, they are **unrepresentable** and the loss goes to `inf`. Without the adapter
it is 3.6 frames/char. `SAMPLES_PER_FRAME = 320`, and the length filter is
`0 < len(labels) * 1.5 < n_samples / SAMPLES_PER_FRAME`.

**d. Phase 1 language routing is free.** Submission ids carry an ISO-639-3 prefix
(`lug_96114` → `lug`) and **all 4,253 of them resolve** — lin 1,866 / sna 1,749 / lug 638, the
same proportions as train. So `lang_from_id()` gets 100% of Phase 1 routing with no model and
no error. `mms-lid-256` stays wired in as the **Phase 2** fallback, since Phase 2 ships no
metadata (§3) and may strip the prefix.

**Data contract as measured:**

| File | Rows | Columns |
|---|---|---|
| `Train.csv` | 38,199 | `id`, `transcription`, `language`, `original_split` |
| `Test.csv` | 4,253 | `ID` — **that is all** |
| `SampleSubmission.csv` | 4,253 | `ID`, `Target` |

`Test.csv` carries no language column and no audio path, so audio has to come from the HF
`test` split — loaded **audio-only**, with `transcription`/`text` dropped on sight by the rules
guard in stages 1 and 3.

**Phase 2 is already open, and we are late to it.** Verified 30 Jul:

```
$ curl -sI https://storage.googleapis.com/waxalphase2/audio.zip
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Length: 762423240                        # 727 MB
Last-Modified: Mon, 27 Jul 2026 07:58:48 GMT
```

The organisers promised the Phase 2 audio "approximately one week before the challenge closes"
and delivered it on 27 Jul. Since **final rankings and prizes are decided on Phase 2, not the
Phase 1 leaderboard**, this is the highest-priority missing input.

Its ID list, **`Test_phase2.csv` (14.7 KB), is missing from our copy and should not be** — it is
on the Zindi **Data** tab, but the bulk-download zip's `manifest-*.json` names only four files
and omits it, so "Download all" doesn't give it to you. Download it separately into
`data/zindi/`. Stages 1 and 3 already glob for it and `wget` the audio zip into
`/kaggle/working` — pull the 727 MB inside a Kaggle notebook, not over a home connection.

**Unknown until we have that file:** whether Phase 2 IDs keep the `lin_`/`sna_`/`lug_` prefix.
"Metadata ... will not be provided" implies not, which would make `mms-lid-256` load-bearing on
the set that actually pays rather than a fallback that never fires.

**Still open:** whether Zindi's server-side scorer matches that starter-notebook reference.
The cheap hedge is to spend two of the five daily slots on the same predictions with and
without punctuation and read the delta off the public leaderboard.

---

## 8. Models evaluated and rejected (so nobody re-litigates them)

| Candidate | Verdict | Why |
|---|---|---|
| [`UBC-NLP/Simba-W`](https://huggingface.co/UBC-NLP/Simba-W) | **No** | Covers lin/sna/lug, but it is Whisper-v3-large **seq2seq**, 1.5B. Cannot take CTC+KenLM shallow fusion — that discards our single biggest lever. Too big to fine-tune on 2×T4, and no per-language metrics published. |
| [`badrex/Ethio-ASR-multilingual-600M`](https://huggingface.co/badrex/Ethio-ASR-multilingual-600M) | **No** | Right base (w2v-bert-2.0), right corpus (WaxalNLP), wrong languages — Amharic/Tigrinya/Oromo/Sidama/Wolaytta only. The card puts lin/sna/lug explicitly out of scope. |
| [`asr-africa/w2v-bert-2.0-*`](https://huggingface.co/asr-africa) (ln / lg / sn) | **No** | Fine-tuned on **1–20 h** of FLEURS/CommonVoice read speech. WAXAL gives us 37–80 h of in-domain spontaneous speech per language. Initialising from a small read-speech fine-tune is a domain step *backwards*, and their vocab doesn't match ours. |
| [DONDO](https://arxiv.org/html/2607.21540) (`KhayaAI/w2v-bert-sna`) | **No** | Genuinely good (Shona 3.02 WER) and Apache-2.0, but Shona **only** — no Lingala, no Luganda — and trained on religious *read* speech. Phase 2 ships no language metadata, so we cannot run a Shona-only model anyway (see §3). |

Net: **raw `facebook/w2v-bert-2.0` remains the init.** Four independent candidates were
checked and none beat it for this task. That is a result, not a gap.

---

## 9. On the existing Whisper work

The Whisper assets in VUKA/BEACON/IMMUNIS do not transfer. Whisper-large zero-shot scores
**93% WER on Lingala and 112% on Shona** on this exact corpus — worse than emitting nothing.
The pipeline shape is reusable; the model is not. This build starts from CTC checkpoints.
