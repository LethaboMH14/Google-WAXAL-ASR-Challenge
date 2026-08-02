# Withdrawn — routing maps for the replaced phase-2 test set

These three maps route the **old** phase-2 test set, which the organisers withdrew on
2026-08-02 (Zindi discussion #34268: "the incorrect Phase 2 test dataset was provided").

They cannot be used. The replacement test set has a completely disjoint id space:

|                | withdrawn set        | corrected set        |
|----------------|----------------------|----------------------|
| clips          | 1,500                | 892                  |
| id shape       | `ID_TBDTM` (5 chars) | `ID_QNYPTX` (6 chars)|
| audio archive  | `audio.zip` (now 404)| `newaudios.zip`      |

Zero ids overlap, so every one of these maps matches nothing against the current audio.
`03_decode_and_submit.py` now raises rather than falling through to the LID when a map named
via `WAXAL_LANG_MAP` covers under half the unlabelled clips — pointing at anything in this
directory will stop the run, which is the intended behaviour.

Kept rather than deleted for one reason: the **router comparison** they encode is still evidence
about the routers themselves. okwija at 0.9917 measured accuracy, mms-closed 0.9792, mms-open
0.9700, asr-conf 0.9658 — those rankings were measured on labelled phase-1 clips and are
unaffected by the phase-2 swap. Reuse the conclusion (okwija wins), regenerate the map.

What they must NOT be used for any more: the "phase 2 is ~95% Luganda" claim. That mix was
measured on the withdrawn audio and says nothing about the corrected set.
