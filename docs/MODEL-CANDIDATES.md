# Open WAXAL checkpoints — survey, access status, and why punctuation decides this competition

*Measured 2026-08-01. Every access status below was verified by an actual `hf_hub_download`, not
by reading a model card — `model_info()` succeeds on gated repos and returns metadata happily,
so "the API answered" is not evidence that we can load the weights.*

## Why this survey exists

The rules permit pretrained models: *"You may use pretrained models as long as they are openly
available to everyone."* They also require disclosure of what we used in the final solution
documentation. This file is that record, and it is also the reason our target moved from
"somewhere above 0.49" to "above the 0.7256 leaders".

## The finding that explains the leaderboard

The organisers publish per-language baselines under `waxal-benchmarking`. Their published test
numbers project onto the competition metric (`0.5(1-WER) + 0.5(1-CER)`, weighted by each
language's share of reference words, because jiwer pools rather than averages) as:

| checkpoint set | WER | CER | projected multi |
|---|---|---|---|
| `mms-300m-waxal-*` | 0.317 | 0.096 | **0.794** |
| `whisper-small-waxal-*` | 0.332 | 0.113 | 0.777 |

But those projections assume the model reproduces punctuation, and **it cannot**. Verified by
downloading `vocab.json` from each:

| checkpoint | tokens | punctuation in vocab |
|---|---|---|
| `waxal-benchmarking/mms-300m-waxal-lin` | 72 | **none** |
| `waxal-benchmarking/mms-300m-waxal-sna` | 51 | **none** |
| `waxal-benchmarking/mms-300m-waxal-lug` | 38 | **none** |

The metric lowercases and does nothing else, so every `.` and `,` in a reference that the
hypothesis omits is one word error plus one character error. Measured on our 900-clip dev set, a
**perfect** transcriber that emits no punctuation scores **0.9367**, not 1.0. Docking that cost
puts `mms-300m-waxal-*` at **~0.731**.

The leaderboard's entire top cluster sits at **0.7206–0.7257**.

That is not a coincidence, and the download counts agree: across the `waxal-benchmarking` org, the
checkpoints for our three competition languages have 5–40× the downloads of the same org's other
languages (`lin` 404, `sna` 434, `lug` 609 vs `mas` 148, `amh` 32, `mlg` 9).

**Conclusion: the leaders are running these checkpoints, and the way past them is punctuation, not
a bigger acoustic model.** A stronger encoder moves WER; punctuation moves a term that everyone
above us is currently paying in full.

## Punctuation statistics (our dev set, n=900, 23,226 ref words)

| | value |
|---|---|
| references ending in `.` | 82.4% overall — lin 64.6%, sna 95.9%, lug 97.8% |
| `.` per utterance | 1.68 (so ~0.86 are sentence-INTERNAL) |
| `,` per utterance | 0.71 |
| `?` / `!` | 2 and 12 occurrences total — negligible, ignore |

Measured value of restoring it, on hypotheses corrupted to a realistic error rate (`punct_probe.py`):

| sim WER | no punct | +trailing `.` | cheap restorer | oracle (all marks) |
|---|---|---|---|---|
| 0.15 | 0.7884 | 0.7996 | 0.7964 | 0.8343 |
| 0.32 | 0.6329 | **0.6412** | 0.6373 | **0.6706** |
| 0.42 | 0.5497 | 0.5566 | 0.5531 | 0.5809 |

Two things follow, and the second one killed a plan:

1. Blindly appending a trailing `.` is worth **+0.008** for zero risk and zero compute.
2. A cheap logistic-regression restorer is **worse than that** (+0.004), despite period F1=0.714.
   Precision is what matters here — a false mark corrupts a word that was otherwise correct — and
   "always append one period" has precision 0.82 by construction, which the classifier could not
   beat. The full oracle is +0.038, so a *good* restorer is still worth chasing, but it has to be
   a real sequence model, not features-and-a-linear-head.

## Access status of every candidate (verified by download)

### Gated — blocked, needs the account holder to accept terms on the model page

`sulaimank` is the single most-downloaded publisher of WAXAL checkpoints and has a complete
punctuation-aware set for exactly our three languages, on the stronger `w2v-bert-2.0` encoder.
All of it is 403 to us right now.

| repo | downloads | what it is |
|---|---|---|
| `sulaimank/w2vbert-lingala-waxal-punct-v2` | 155 | punct-in-vocab, lin |
| `sulaimank/w2vbert-shona-waxal-punct-v2` | 288 | punct-in-vocab, sna |
| `sulaimank/w2vbert-luganda-waxal-punct-v2` | 145 | punct-in-vocab, lug |
| `sulaimank/w2vbert-{lingala,shona,luganda}-waxal` | 460 / 410 / 393 | same, no punct |
| `sulaimank/waxal-punct-restorer` | 122 | `XLMRobertaForTokenClassification` — a ready-made restorer |
| `sulaimank/omniASR-CTC-300M-waxal` | — | one CTC model for all languages |
| `douyeszn/w2vbert-sna-waxal-aug` | 18 | the only w2v-bert sna we found outside sulaimank |

Note for the rules: gated-but-free is a grey area against *"openly available to everyone"*. Anyone
can click through, so we read it as permitted — but it must be disclosed, and we should not build
the whole solution on it without a fallback that is unambiguously open. The lineup below is that
fallback.

### Open — usable today

| repo | arch | lang | punctuation in vocab |
|---|---|---|---|
| `keystats/lingala-xlsr-waxal-finetuned` | Wav2Vec2ForCTC | lin | **`! " ' , - . : ; ?`** |
| `douyeszn/w2vbert-lug-waxal-aug` | Wav2Vec2BertForCTC | lug | **`! " ' , - . : ; ?`** |
| `dhasmana/WAXAL-lug-ful-w2v-bert-2.0` | Wav2Vec2BertForCTC | lug | `.` only |
| `douyeszn/w2vbert-lin-waxal-aug-ft` | Wav2Vec2BertForCTC | lin | `'` only |
| `Mubarak127/waxal-whisper-large-v3-sna_asr` | Whisper | sna | native (BPE) |
| `cdli/whisper-large-v3_finetuned_ugandan_luganda_waxal_7...` | Whisper | lug | native (BPE) |
| `ElizabethMwangi/whisper-large-v3-luganda-waxal-1` | Whisper | lug | native (BPE) |
| `VinyVan/xlsr-luganda-waxal` | Wav2Vec2ForCTC | lug | `'` only |
| `waxal-benchmarking/mms-300m-waxal-{lin,sna,lug}` | Wav2Vec2ForCTC | all | none |
| `Okwija/waxal-lid-lin-sna-lug` | Wav2Vec2ForSequenceClassification | LID | — |

A fully-open, punctuation-capable lineup therefore already exists:

- **lin** → `keystats/lingala-xlsr-waxal-finetuned`
- **sna** → `Mubarak127/waxal-whisper-large-v3-sna_asr` (Whisper emits punctuation natively)
- **lug** → `douyeszn/w2vbert-lug-waxal-aug`

`Okwija/waxal-lid-lin-sna-lug` matters separately: phase 2 ships audio with **no language
metadata**, so language ID is part of the task. It is a purpose-built classifier for exactly our
three languages and should be measured against our existing 97.7% open-set router.

## What is NOT settled

Every projection above is arithmetic on other people's published numbers, on their test split, and
not one of these checkpoints has yet been run through our own dev harness. That is what the
bakeoff kernel is for. Nothing here gets submitted on the strength of a table.
