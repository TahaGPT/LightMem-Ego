# PROJECT.md — On-Device Egocentric AI Memory Assistant

> **Purpose of this document**: This is the single source of truth for this project. It is written to be handed to any AI assistant (Claude, GPT, Gemini, local LLM, etc.) as context so that assistant can immediately understand the full scope, architecture, constraints, and current status without needing prior conversation history. Update this file as decisions change.

**Last updated**: 2026-09-26
**Status**: Pre-implementation / architecture & research phase
**Owner**: [fill in]

---

## 1. One-line summary

A fully offline, phone-local AI system that continuously perceives the world through an egocentric (glasses-style) camera and microphone, builds a searchable long-term memory of what it has seen and heard, and answers real-time and historical questions via voice or text — with zero external API calls.

---

## 2. Vision / inspiration

Modeled conceptually on Meta Ray-Ban-style smart glasses ("Hey Meta" wake word, voice-driven AI assistant), but with two hard constraints Meta's own product does not have:

1. **100% on-device inference** — no cloud API calls, ever, for any part of the pipeline (perception, reasoning, storage, or retrieval).
2. **Persistent long-term memory** — not just live Q&A, but the ability to query things observed minutes, days, or months in the past.

**Important known reality**: Meta's own Ray-Ban Meta glasses do **not** achieve this — most of their heavier AI queries are routed to the cloud, and the glasses themselves have no meaningful on-device compute. This project is explicitly harder than what Meta ships today. See Section 9 (Reality Check) before committing to scope.

---

## 3. Target use cases (functional requirements)

### 3.1 Live / real-time queries (answered from current camera+mic frame, no memory lookup needed)
- "What is this?"
- "What am I looking at?"
- "What's written on this paper?" (OCR)
- General live visual question-answering

### 3.2 Short-term memory queries (minutes to hours ago)
- "Where did I place my phone?"
- "What did I just say / hear?"

### 3.3 Long-term memory queries (days to months/years ago)
- "What happened in the meeting three days ago?"
- "What was the code written on the whiteboard three months ago?"
- "What was the password I wrote on the paper for my Gmail a year ago?"

> ⚠️ **Security note on 3.3 last example**: any pipeline that stores credentials/PII captured incidentally must treat that data as sensitive by default. See Section 8 (Security & Privacy) — this is a mandatory design constraint, not optional polish.

### 3.4 Interaction model
- **Wake word**: e.g. "Hey [assistant name]" — activates the pipeline from idle.
- **Stop word**: e.g. "Stop [assistant name]" — deactivates active listening/processing.
- **Output modes**: voice (TTS) or text, user-selectable.
- **Input modes**: continuous ambient camera + mic (gated, not literally continuous inference — see Section 5), plus explicit voice questions after wake word.

### 3.5 Explicit non-goals (out of scope for v1)
- Cloud fallback for "hard" queries (violates the no-external-API constraint by design).
- Real smart-glasses hardware integration in v1 — build and validate on phone camera/mic first (see Section 9 on glasses hardware reality).
- Multi-user / shared-device support.
- Real-time speaker diarization at production quality (flagged as "partially achievable," Section 9).
- Literal full-fidelity video archival ("rewind to any exact second") — this system stores a **lossy, sampled, captioned memory**, not raw footage.

---

## 4. High-level architecture

**Core design principle**: Nothing runs continuously except near-zero-power sensors. Heavy models (VLM, ASR) are event-triggered, not clocked. This is the single most important architectural decision in this project — a naive "run the VLM on every frame" design is not viable on phone battery/thermal budgets (see Section 9).

```
TIER 0 — Always-on (≈0 battery impact)
  ├─ Wake-word spotter        (listens 24/7 for activation phrase)
  ├─ Voice Activity Detector  (gates whether ASR should run at all)
  └─ Motion / scene-change detector (gates whether a new keyframe is worth captioning)
        │
        ▼ (event fires)
TIER 1 — Small model (fast, cheap, kept warm)
  ├─ Intent classifier: live visual Q&A? memory recall? chit-chat?
  └─ Conversational front-end / response drafting
        │
        ▼
TIER 2 — Heavy models (loaded only on demand)
  ├─ VLM: deep visual understanding, OCR, reasoning over current frame
  └─ ASR: full transcription of speech (only when VAD/query triggers it)
        │
        ▼
TIER 3 — Background indexer (runs opportunistically — idle time / charging)
  ├─ Keyframe captioning → embeddings → vector DB insert
  └─ Nightly "consolidation" pass: compress a day's raw captures into
      summarized daily/weekly embeddings (inspired by biological memory
      consolidation during sleep)
```

### 4.1 Data flow for a live query
```
User speaks wake word
  → Tier 0 wake-word spotter fires
  → Tier 1 small model classifies intent (live vs. memory vs. chit-chat)
  → IF live: grab current camera frame → Tier 2 VLM answers directly
  → IF memory: query vector DB for relevant embeddings/timestamp window
      → retrieve matching captions/thumbnails
      → Tier 2 VLM/LLM synthesizes final answer from retrieved context
      → (optional) re-run heavy VLM OCR pass on retrieved thumbnail for
        precision-critical answers (e.g. reading exact text)
  → Response generated as text
  → TTS renders voice output (if voice mode selected)
  → User says stop word → pipeline returns to Tier 0 idle state
```

---

## 5. Component specification

### 5.1 Inference runtime (on-device engine)

| Choice | Role | Rationale |
|---|---|---|
| **Cactus** (primary recommendation) | Unified mobile inference engine for LLM + VLM + ASR + embeddings | Purpose-built for exactly this multimodal stack; native iOS/Android/Flutter/React Native SDKs; NPU acceleration; INT4/INT8 quantization support; first-class support for Gemma, Qwen, LFM2, Whisper, Moonshine, Parakeet, Nomic Embed. Caveat: small-vendor SDK, less battle-tested than Google/Meta tooling — validate maturity during Phase 0. |
| **LiteRT-LM / MediaPipe LLM API** (fallback / Android-native alt) | Google's official on-device runtime | Best NPU integration on Android/Pixel; co-designed with Qualcomm/MediaTek/Samsung for Gemma 3n specifically. Less unified — requires bolting on your own vector DB and wake-word layer. |
| **ExecuTorch** (fallback / max control) | Meta's production-grade edge inference framework | Widest backend coverage, most control, more manual integration work. |
| **llama.cpp** (prototyping only) | Broadest community support | Good for early benchmarking; rougher mobile vision/multimodal bindings — not recommended for shipping. |
| **Ollama** | ❌ Not used | Desktop/server-first, not a mobile deployment target. |

### 5.2 Model manifest

| Role | Model | Size / precision | Always-on? | Notes |
|---|---|---|---|---|
| Wake-word detection | Porcupine (Picovoice) or openWakeWord | KBs–low MBs | ✅ Yes, 24/7 | Only component that should never sleep. |
| Voice Activity Detection | Silero VAD | ~1MB | ✅ Yes, 24/7 | Gates ASR invocation. |
| Scene-change / motion gate | Lightweight frame-diff / motion heuristic (not a full model) | Trivial | ✅ Yes, 24/7 | Gates keyframe capture — decides "is this moment worth captioning." |
| Continuous passive tagger | **Moondream2** (0.5B, or int4 2B variant) | ~0.5–2B, INT4/INT8 | Event-triggered | SigLIP + Phi-1.5 based; small enough for frequent low-cost captioning. |
| Heavy on-demand VLM | **Gemma 3n (E2B / E4B)** | 5B/8B raw params, runs in ~2–3GB RAM via MatFormer elastic architecture | On-demand (wake word / query) | Natively multimodal (image, audio, video, text); MobileNet-V5 vision encoder; MatFormer lets it also serve as the "small model" via elastic slicing — potentially collapsing Tier 1 and Tier 2 into one base model. |
| Small conversational/intent model | Gemma 3n small slice (MatFormer) OR separate ~0.5–1B model (Qwen2.5-0.5B / LFM2-350M) | Small, always-resident | ✅ Kept warm | Decision pending: use MatFormer slicing of the same weights vs. a fully separate tiny model — evaluate both in Phase 0. |
| ASR | Whisper-tiny/base (quantized) or Moonshine | Small, INT8 | Event-triggered (VAD gated) | Only runs when audio activity detected. |
| Text embeddings | Nomic Embed | Small | On write / on query | For captions, transcripts, and query text. |
| Image embeddings | Reused from VLM's own SigLIP-derived vision encoder | N/A | On write | Do **not** run a separate CLIP pass — reuse the encoder already inside Moondream/Gemma 3n. |
| Vector DB | sqlite-vec or ObjectBox (HNSW-based) | N/A | Local storage | Holds embeddings + timestamp/geo metadata + pointer to compressed thumbnail (not raw frames). |
| TTS | Native OS TTS or Piper (offline) | N/A | On demand | Don't over-engineer; mature solved problem. |

### 5.3 Model efficiency / compression strategy

Do **not** rely on quantization alone. Layered strategy, in priority order:

1. **MatFormer elastic slicing** (primary lever) — Gemma 3n was jointly trained to support extracting smaller submodels without post-hoc approximation error. Use this to get your "small model" from the same weights as your "big model" wherever possible.
2. **INT4/INT8 quantization on top** (AWQ or GPTQ style) — for final memory footprint reduction. Expect small-but-real quality loss; budget for evaluation (Section 10).
3. **Speculative decoding** (optional, latency only) — provably lossless acceleration technique; use only if Phase 0 benchmarking shows latency (not RAM) is the bottleneck. Does not reduce memory footprint — requires both a draft and target model resident.
4. **Native low-bit training (BitNet-style)** — watch this space; if a BitNet-native variant of your chosen base model becomes available, it avoids quantization's post-hoc degradation entirely (trained at 1.58-bit from scratch rather than rounded after training). Not usable on Gemma 3n/Moondream today unless such a variant is released.
5. **Pruning (SparseGPT/Wanda)** — deprioritized for this project. Their strongest "near-zero loss" results are demonstrated at large scale (66B+ params); the small 2–4B models used here have less redundancy to safely cut, and unstructured sparsity often doesn't translate into real speedups on current mobile NPUs without explicit kernel support.
6. **Knowledge distillation** — not something to build in-house for v1; both Gemma 3n and Moondream already benefit from distillation-style training upstream. Revisit only if a custom smaller model becomes necessary.

---

## 6. Memory & storage architecture

### 6.1 Short-term memory
- In-RAM ring buffer holding the last **N minutes** of keyframes + transcript snippets.
- Purpose: answer "what did I just see/hear" without touching the vector DB.
- No persistence — cleared on rotation.

### 6.2 Long-term memory (persistent, on-device)
Each memory entry stored with:
- `timestamp`
- `geolocation` (optional, user-configurable)
- `modality` (visual / audio / text-OCR)
- `caption_text` (from Tier 2/3 captioning pass)
- `embedding_vector` (from reused vision/text encoder)
- `thumbnail_path` (compressed low-res image, NOT full-resolution raw frame)
- `sensitivity_flag` (boolean/category — see Section 8)
- `source_tier` (which model/pipeline stage produced this entry)

### 6.3 Consolidation pipeline
- Nightly (or on-charger-idle) background job.
- Compresses a day's raw keyframe captions/embeddings into summarized daily embeddings — analogous to human sleep-based memory consolidation.
- Reduces long-term storage growth rate and improves retrieval relevance over long horizons.

### 6.4 Deep re-inspection on demand
- When a query needs fine detail (e.g., "what was the exact code on the whiteboard"), retrieve the stored thumbnail for that timestamp and **re-run it through the heavy VLM's OCR path** rather than trusting the original lightweight caption — the fast Tier 3 tagger is optimized for throughput, not precision.

### 6.5 Retention policy
- [Decision pending] Define max storage budget, auto-deletion/rolling-window policy for non-flagged low-importance memories, and user-facing controls to manually delete/export memories.

---

## 7. Wake-word / stop-word interaction spec

- Activation phrase: [define exact phrase]
- Deactivation phrase: [define exact phrase]
- Behavior on activation: Tier 1 model wakes, begins listening for a query; if no query within [X seconds], auto-return to idle.
- Behavior on deactivation: immediately halt any in-progress Tier 2 inference if safe to do so; return to Tier 0 idle state.
- False-positive handling: [decision pending — confirmation chime? visual indicator?]

---

## 8. Security & privacy (mandatory design constraints, not optional)

1. **No external network calls, ever** — this is a hard architectural invariant, not just a preference. Any future feature must be checked against this before implementation.
2. **Sensitive-content detection**: pipeline must flag captured content that looks like credentials, PII, financial info, or identity documents (e.g. "password written on paper" scenario from Section 3.3) and route it to **encrypted storage** (Android Keystore / iOS Secure Enclave-backed encryption) rather than plain embedding storage.
3. **OCR reliability caveat**: handwriting OCR errors are a real risk for anything security-critical (e.g., recalling a password verbatim). Do not present OCR'd sensitive text with false confidence — surface uncertainty to the user.
4. **Legal/consent consideration**: continuous audio/video capture of third parties raises consent and wiretapping-law questions in many jurisdictions (some require two-party consent for audio recording). This needs real legal review before any deployment beyond personal single-user testing — not covered by this document.
5. **User data ownership**: all data stays device-local; user should have visibility into and control over (view/export/delete) what has been stored.

---

## 9. Reality check (read before committing engineering time)

### 9.1 Achievable today
- Wake-word-triggered live Q&A ("what is this," "what's written on this paper") with decent latency/quality — proven pattern (e.g. Envision, Seeing AI apps already ship this).
- Short-term recall (minutes–hours) via ring buffer.
- Fully offline voice in/out.
- Reasonably good OCR on clear printed text in good lighting.

### 9.2 Partially achievable
- Multi-day/multi-week memory recall — works, but is a **lossy, sampled** memory (captions + embeddings of keyframes), not literal video replay. Recall quality depends entirely on whether something was captured and captioned well at that exact moment.
- Meeting summaries from days ago — topic-level summaries realistic; verbatim quotes/exact numbers/speaker diarization are weak on-device.
- Reading back a password written on paper a year ago — possible only if that exact frame was captured and OCR'd correctly at the time; handwriting OCR errors are a real risk (see Section 8.3).
- Overall answer quality — local 2–4B models will noticeably underperform cloud-scale LLMs on synthesis/reasoning. Expect shorter, less nuanced answers than cloud LLM equivalents.

### 9.3 Not achievable with today's hardware
- True 24/7 full-resolution continuous video+audio inference on a phone/glasses without severe battery drain and thermal throttling within roughly an hour. This is why the event-triggered architecture (Section 4) is mandatory, not optional.
- Running any of this compute **on glasses hardware itself** — no consumer smart glasses today have compute for even a 1B model on the glasses; it must run on the paired phone. Notably, even Meta's own Ray-Ban Meta glasses route most heavy AI queries to the cloud rather than processing on-device.
- Perfect verbatim recall of arbitrary past details as if it were a searchable video archive — would require unbounded storage and full-archive video search per query, infeasible on a phone in real time. This system is a diary that samples and summarizes, not a rewind button.

---

## 10. Benchmarking & evaluation plan

### 10.1 Phase 0 — device feasibility spike
- Measure on target phone: tokens/sec, time-to-first-token, battery %/minute of active inference, thermal throttling onset time.
- This number sets the entire duty-cycle budget for the rest of the system.
- Methodology reference: base protocol on **MLPerf Mobile** (MLCommons) methodology for latency/power measurement, even without formal submission.

### 10.2 VLM answer quality
- **VQAv2** — general open-ended visual QA quality.
- **GQA** — compositional/spatial reasoning.
- **TextVQA** — reading text in images (directly relevant to "what's written on the paper" use case).
- **POPE** — object hallucination rate (binary presence probing with random/popular/adversarial negative sampling) — critical for measuring how often the memory system invents objects/text that weren't there.

### 10.3 Quantization / compression quality
- Track perplexity delta on held-out text (WikiText/C4-style) before vs. after each compression step (quantization, pruning, slicing) — standard practice from the compression literature (Section 11).

### 10.4 Retrieval quality
- Recall@K and nDCG for "did vector search surface the right memory."
- End-to-end exact-match/F1 for "did the final answer contain the right fact" (borrowed from open-domain QA eval harnesses, e.g. Natural Questions/TriviaQA-style scoring).

### 10.5 Egocentric-specific evaluation
- Consider evaluating against **Ego4D**'s "episodic memory" benchmark task (querying the past from first-person video) before trusting the custom pipeline's real-world recall quality.

---

## 11. Research bibliography (organized by component)

### Vision-language modeling
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021 — arXiv:2103.00020
- Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (SigLIP), 2023 — arXiv:2303.15343
- Devvrit et al., "MatFormer: Nested Transformer for Elastic Inference," 2023 — arXiv:2310.07707
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022 — arXiv:2212.04356

### Quantization
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," 2022 — arXiv:2208.07339
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," 2022 — arXiv:2210.17323
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," 2023 — arXiv:2306.00978
- Xiao et al., "SmoothQuant" (referenced 2022/2023)

### Native low-bit training (quantization alternative)
- Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models" (original BitNet)
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," 2024 — arXiv:2402.17764
- Ma et al. (Microsoft Research), "BitNet b1.58 2B4T Technical Report," 2025 — arXiv:2504.12285

### Pruning (quantization alternative)
- Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot," 2023 — arXiv:2301.00774
- Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (Wanda), 2023 — arXiv:2306.11695

### Lossless inference acceleration
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," 2023 — arXiv:2211.17192

### Long-context / streaming inference
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM), 2023 — arXiv:2309.17453

### Retrieval and long-term memory architecture
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020 — arXiv:2005.11401
- Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" (HNSW), 2018 — arXiv:1603.09320
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," 2023 — arXiv:2304.03442 (memory stream: recency + importance-weighted retrieval — directly applicable to Section 6.3 consolidation design)
- Packer et al., "MemGPT: Towards LLMs as Operating Systems," 2023 — arXiv:2310.08560 (OS-style paged memory pattern underlying the Tier 1/Tier 2 split)
- Also worth tracking: "Mem0" (Chhikara et al.) — more recent scalable persistent-memory architecture for LLMs.

### Egocentric video understanding (the actual data domain)
- Grauman et al., "Ego4D: Around the World in 3,000 Hours of Egocentric Video," CVPR 2022 — arXiv:2110.07058 (includes an "episodic memory" benchmark task directly analogous to Section 3.3 use cases)
- Also search: Meta's "Project Aria" research program for egocentric hardware/sensor-side literature (distinct from Ego4D's dataset/benchmark focus).

### Knowledge distillation
- Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network," 2015 (foundational reference)

### Evaluation benchmarks
- Goyal et al., "Making the V in VQA Matter" (VQAv2), 2017
- Hudson & Manning, "GQA," 2019
- Singh et al., "TextVQA," 2019
- Li et al., "Evaluating Object Hallucination in Large Vision-Language Models" (POPE), 2023 — arXiv:2305.10355
- MLCommons, "MLPerf Mobile" benchmark suite/methodology (industry standard for on-device latency/power measurement, not a single paper)

---

## 12. Phased build roadmap

1. **Feasibility spike** — benchmark chosen runtime + Gemma 3n E2B on target phone (tokens/sec, time-to-first-token, battery %/min, thermal throttling onset).
2. **MVP: push-to-talk, no memory** — wake word → grab one frame → VLM answers → TTS. Validates latency/quality baseline.
3. **Short-term ring buffer** — last N minutes in RAM. Enables "where did I just put my phone."
4. **Event-triggered indexing** — scene-change/motion gating → fast captioner → embeddings → vector DB. Gets recall from minutes to days.
5. **Long-horizon storage** — captions + embeddings + thumbnails only (not raw frames); nightly consolidation pass.
6. **Deep re-inspection on demand** — pull stored thumbnail for fine-detail queries, re-run through heavy VLM's OCR path.
7. **Security hardening** — sensitive-content detection + encrypted storage routing (Section 8).
8. **Power/thermal tuning** — adaptive duty cycling, NPU offload, dynamic frame-rate scaling by battery state.

---

## 13. Open decisions / TODO

- [ ] Confirm exact wake word / stop word phrases and false-positive handling behavior.
- [ ] Decide: MatFormer-sliced single base model vs. fully separate small+large models for Tier 1/Tier 2.
- [ ] Define storage retention policy and user-facing memory management controls.
- [ ] Legal review of audio/video consent requirements for target deployment jurisdictions.
- [ ] Evaluate Cactus SDK maturity vs. LiteRT-LM/MediaPipe during Phase 0 before committing.
- [ ] Decide on target phone hardware baseline for all benchmarking (Section 10.1).
- [ ] Investigate whether a BitNet-native variant of the chosen base model becomes available (Section 5.3, item 4).

---

## 14. Glossary

- **Tier 0/1/2/3**: the four-stage pipeline hierarchy defined in Section 4, from always-on sensors to background indexing.
- **MatFormer / elastic slicing**: training technique allowing one model's weights to be sliced into multiple smaller, independently-competent submodels (arXiv:2310.07707).
- **PTQ (Post-Training Quantization)**: reducing weight precision after training (e.g. GPTQ, AWQ) — introduces approximation error.
- **QAT-adjacent / native low-bit training**: training a model natively at low precision from scratch (e.g. BitNet) rather than quantizing afterward.
- **Event-triggered capture**: capturing/processing a frame only when a motion/scene-change gate fires, not on a fixed clock — the core mechanism that makes battery life viable.
- **Consolidation**: background compression of a day's raw memory entries into summarized long-term embeddings, inspired by biological sleep-based memory consolidation.
- **Keyframe**: a single representative captured frame selected by the scene-change gate, stored with its caption/embedding/thumbnail — not continuous video.

---

## 15. Panel Defense Prep — technique/limitation → research paper mapping

> Purpose: every architectural choice in this document should be traceable to a paper, and every claimed limitation should be traceable to a paper or to original empirical work performed by the project team. This section is the traceability matrix for that.

### 15.1 Technique → paper (what backs each architectural choice)

| Component | Technique | Paper | What it backs |
|---|---|---|---|
| Tier 0 gate | Event-triggered capture instead of continuous inference | Phase Matters — arXiv:2606.27906 | Justifies gating instead of running inference on every frame |
| Tier 1/Tier 2 split | Small-model-routes-to-large-model cascade | FrugalGPT — arXiv:2305.05176; "LLM-SLM Collaboration...Pipelines and Triggers" — arXiv:2402.05621 | Formal cascade/trigger framework behind the Tier 1→Tier 2 handoff |
| One base model, two effective sizes | MatFormer elastic slicing | arXiv:2310.07707 | Deriving Tier 1 and Tier 2 from the same Gemma 3n weights |
| Vision-language backbone | SigLIP encoder + Gemma 3n | arXiv:2303.15343; Gemma 3n Technical Report (Google DeepMind) | Base model architecture |
| Compression | AWQ INT4 quantization | arXiv:2306.00978 | Primary compression method |
| Memory schema | Working + episodic dual memory | VLM² — arXiv:2511.20644 | Precedent for ring-buffer (working) + vector-DB (episodic) split |
| Consolidation job | Recency+importance memory stream; long-term compression | Generative Agents — arXiv:2304.03442; StreamMeCo — arXiv:2604.09000 | Nightly consolidation design |
| Retrieval | RAG formalism + HNSW ANN search | arXiv:2005.11401; arXiv:1603.09320 | Retrieve-then-generate backbone |
| OCR gating | Reading-activity detection before running OCR | Reading Recognition in the Wild — arXiv:2505.24848 | Why Tier 0 should detect "is the user reading" before invoking heavy OCR |

### 15.2 Limitation → paper (what backs each claimed constraint)

| Limitation | Backing paper(s) | Concrete figure to present |
|---|---|---|
| RAM footprint | Gemma 3 QAT report (Google); pytorch/gemma-3-12b-it-INT4 model card | Gemma 3n-E2B: 19.3GB (FP16) → 3.2GB (INT4); Gemma-3-12B peak memory 24.5GB→8.68GB (65% reduction) |
| Compute / CPU overload | OnePlus 13R study — arXiv:2507.08505 | 600–800% CPU load (8 cores), GPU/NPU at 0% in 3 of 4 configs |
| Battery drain | arXiv:2507.08505 | ~8 hrs (CPU-only) vs. ~2 days (GPU-offloaded) at 1 query/min, 5000mAh cell |
| Thermal throttling | arXiv:2507.08505; Phase Matters — arXiv:2606.27906 | 88–95°C (CPU-bound) vs. 60°C (GPU-offloaded); NPU offload = 10.47°C lower steady-state |
| Latency | arXiv:2507.08505 Table 2; Gemma 3n LiteRT-LM benchmark table | 22–174s end-to-end depending on config; decode ≈16 tok/s regardless of CPU/GPU |
| Quantization accuracy loss | AWQ arXiv:2306.00978; Gemma-3-12B card; Unsloth Gemma-4 QAT docs | MMLU 71.51→68.96 (INT4); naive Q4_0 conversion = 70.2% top-1 vs. 85.6% with proper conversion |
| OCR reliability on egocentric footage | "An Evaluation of OCR on Egocentric Data" — arXiv:2206.05496; Reading Recognition in the Wild — arXiv:2505.24848 | OCR measurably degrades on egocentric footage vs. static photos |
| Hallucination / false memory recall | POPE — arXiv:2305.10355 (general); EgoMemReason — arXiv:2605.09874; EgoMonth — arXiv:2608.13113 (long-horizon egocentric) | Use POPE's probing methodology as the hallucination-rate test protocol |
| Meeting summarization / diarization accuracy | "Fast and Robust On-Device Speaker Diarization" — arXiv:2606.08505; "On the Limitations of Speaker Diarization" (Amorim, *Expert Systems*, 2026) | On-device acceleration raises DER 0.075→0.113 on in-the-wild audio (VoxConverse) |
| NPU underutilization by current frameworks | arXiv:2507.08505; arXiv:2606.27906 | NPU unused in 3/4 tested configs; phase-dependent when used (1.64× prefill, 1.18× decode) |

### 15.3 Confirmed research gaps — no paper exists; original empirical work required

For each of these, state explicitly to any audience: *"No existing literature addresses this at our scale — we ran our own measurement."* Show methodology and numbers.

| Gap | Why no paper covers it | Methodology to generate the figure yourself |
|---|---|---|
| Long-term storage growth (months–years of captions+embeddings+thumbnails) | No paper models a year of personal lifelog storage at this project's specific capture rate | Calculate: (embedding dim × bytes/float × entries/day × 365) + (thumbnail size × entries/day × 365). Run the event-triggered gate on one real day of footage, count actual keyframes, extrapolate. |
| Retrieval quality as the vector DB scales to months of entries | HNSW's paper covers generic ANN scaling, not personal egocentric memory at this scale | Build a 2–4 week test set with known ground-truth answers, measure Recall@K and nDCG; explicitly caveat that a full year is untested. |
| Wake-word false-accept/false-reject rate for the chosen phrase and environment | Porcupine/openWakeWord publish general numbers, not for this project's exact phrase/device/noise conditions | Record N trials (e.g. 100 quiet + 100 noisy), compute false-accept/false-reject rate directly. |
| Intent-routing accuracy of the Tier 1 model (live vs. memory vs. chit-chat) | Cascade/routing papers validate the concept, not this project's specific 3-way classification | Build a labeled test set (50–100 examples across the three categories), measure accuracy/confusion matrix. |
| Consolidation information loss (nightly summarization vs. raw captured day) | StreamMeCo addresses compression conceptually, not this pipeline's specific numbers | Run consolidation on a real captured day, test recall on raw vs. consolidated versions with the same question set, report the accuracy delta. |
| Real smart-glasses camera/mic API access constraints | Product/SDK reality, not an academic research question | Pull directly from current developer documentation for the target glasses platform; present as a stated product constraint, not a data point. |
| Consent/legal exposure for continuous audio/video capture of bystanders | Legal scholarship exists but isn't "research-backed" in the ML sense | Flag explicitly as out-of-scope for technical evaluation, requiring separate legal review — do not force a technical citation onto a legal question. |

### 15.4 Suggested reading order (time-constrained prep)

1. **arXiv:2507.08505** (OnePlus 13R) — richest single source of real compute/battery/thermal/latency figures together.
2. **arXiv:2606.27906** (Phase Matters) — NPU-specific follow-up; the "why NPU offload matters" story with numbers.
3. **EgoMemReason (arXiv:2605.09874) + EgoMonth (arXiv:2608.13113)** — defines this project's exact problem as an open research question; strongest "recognized gap" framing.
4. **AWQ (arXiv:2306.00978) + Gemma 3 QAT report** — the quantization technique and its measured cost.
5. **POPE (arXiv:2305.10355)** — copy its methodology directly as this project's own hallucination-testing protocol.
6. **FrugalGPT (arXiv:2305.05176)** — the one-paragraph justification for the two-model cascade architecture.

---

*End of Project.md — keep this file updated as the single context source for any AI assistance on this project.*
