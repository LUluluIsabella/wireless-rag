## Retrieval Evaluation

### Experimental Setup

- Corpus: 3 Wi-Fi 6 technical documents
- Documents: Cisco, ZTE, Spirent
- Total Chunks: 156
- Embedding Model: intfloat/multilingual-e5-small
- Dense Index: FAISS IndexFlatIP
- Sparse Retrieval: BM25
- Hybrid Retrieval: Reciprocal Rank Fusion (RRF)
- RRF k: 60
- Retrieval Top-K: 5

### Evaluation Dataset

- Development Set: 10 questions (8 answerable)
- Test Set: 30 questions (26 answerable)
- Unanswerable Test Questions: 4

Note: Results are preliminary because the candidate Gold
annotations have not undergone complete manual review.

### Test Set Results

| Method | Hit@5 | MRR@5 | Hits |
|---|---:|---:|---:|
| Dense | 0.7308 | 0.4654 | 19/26 |
| BM25 | 0.7692 | 0.5833 | 20/26 |
| Hybrid | 0.8462 | 0.7244 | 22/26 |

### Discussion

Hybrid Retrieval achieved the best performance on both
Hit@5 and MRR@5.

Compared with Dense Retrieval, Hybrid improved Hit@5
by 11.54 percentage points and MRR@5 by approximately
55.7% in relative terms.

Compared with BM25, Hybrid improved Hit@5 by 7.70
percentage points and MRR@5 by approximately 24.2%.

These preliminary findings suggest that combining
semantic and lexical retrieval through RRF improves
evidence retrieval quality on the evaluated Wi-Fi 6
technical questions.

However, ranking regressions were observed in individual
queries. These cases are documented in ISSUE-003.

### Limitations

1. The evaluation corpus contains only three documents.
2. The test set contains 26 answerable questions.
3. Gold annotations require final manual verification.
4. Unanswerable questions are evaluated separately.
5. Some PDF chunks contain layout and table extraction noise.
6. BM25 and Hybrid latency measurements include initial
   Jieba initialization overhead.

## Generation Evaluation (2026-10-24)

### Setup and Method

- Dataset: `evaluation/generation_test.jsonl`, a fixed subset of nine questions from `evaluation/test.jsonl` (five answerable: `test_001`, `test_006`, `test_012`, `test_020`, `test_021`; four unanswerable: `test_027`–`test_030`).
- Pipeline: Hybrid retrieval, Top 5 chunks, followed by the configured LLM via `generate_answer()`.
- Each output records the generated answer, retrieved chunks, citations, and mapped sources in `evaluation/generation_results.json`.
- Citation validity is checked programmatically. Core claim support and appropriate abstention were reviewed manually against the retrieved evidence and expected answerability.

### Preliminary Results

| Measure | Result | Interpretation |
|---|---:|---|
| Execution success | 9/9 | All nine retrieval-and-generation calls completed without reported errors. |
| Invalid citations detected in original run | 0 | All 28 extracted citation IDs mapped to available sources; this is not a claim-level validity measure. |
| Answerable questions with supported core conclusions | 5/5 | Manual question-level check; `test_012` contains an overgeneralized description. |
| Correct abstentions on unanswerable questions | 4/4 (100%) | No unsupported numbers, prices, passwords, or precise RTT were invented. |
| Incorrect abstentions on answerable questions | 0/5 | All five questions received substantive answers. |

These results are **preliminary and small-sample**. The 5/5 core-conclusion observation is **not** a strict sentence-level Citation Support Rate or an overall faithfulness score. The four unanswerable cases were judged by inspecting their `generated_answer` content; their `human_evaluation` fields in the original result file were not populated by the evaluation script.

### Case Review

- `test_001`: The answer of nine 26-tone RUs in a 20 MHz channel is supported by Spirent and ZTE chunks.
- `test_006`: Higher SNR for 1024-QAM is supported by ZTE and Cisco; Cisco gives an approximate additional 6 dB requirement.
- `test_012`: BSS Color and differentiated detection thresholds are supported. The answer's suggestion that neighboring BSSs need not defer is too absolute: transmission opportunities still depend on detection thresholds and interference conditions.
- `test_020`: The minimum RU size of 106 subcarriers for combined MU-MIMO/OFDMA is supported by Spirent.
- `test_021`: The general low-bandwidth OFDMA versus high-bandwidth/full-buffer MU-MIMO comparison is supported by Spirent, with deployment qualifications.
- `test_027`–`test_030`: All four answers explicitly stated insufficient evidence and declined to invent requested details.

### Citation Validation Regression Tests

After reviewing the generation results, `extract_citations()` was updated to support ordinary, repeated, and ranged citations (such as `[S1]–[S5]` and `[S1-S5]`). The user ran `python -m pytest tests/test_generator.py -v` and reported **7/7 tests passed**. The tests covered single/multiple/ranged citations, deduplication, extraction of `[S99]`, and a mocked LLM response confirming that an invalid `[S99]` is recorded and the original answer is withheld.

This validates the tested local parsing and rejection paths only. The 28-citation count above comes from the **original** generation run, before the parser update; the original JSON was not regenerated. Ranged-notation parsing previously missed intermediate citations (for example, `[S1]–[S5]` was read as `S1`, `S5`). Malformed and oversized ranges require a documented policy and further tests.

### Outstanding Work and Limitations

1. Test at least one prompt-injection instruction embedded in a simulated retrieved chunk, using the real LLM. The system prompt contains defensive instructions, but effectiveness has **not yet been evaluated**.
2. Add a dedicated malformed-citation test suite; ensure citations are not only syntactically valid but also support each associated claim.
3. Perform a stricter claim-by-claim support audit if reporting a formal Citation Support Rate.
4. Review candidate Gold annotations, especially retrieval misses `test_004`, `test_013`, `test_017`, and `test_026`; do not tune retrieval against the frozen test set.
5. Retain corpus and index version identifiers for reproducibility. Results from nine hand-picked cases must not be generalized to broad model reliability.
