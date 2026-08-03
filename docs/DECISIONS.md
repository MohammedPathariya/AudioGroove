# AudioGroove Decisions

This file records decisions that affect architecture, experiments, evaluation, and deployment. Update it when a decision changes. Do not silently replace an existing decision.

## D-001: Laptop-first development

- **Decision:** The primary development machine is an Apple MacBook Air M1 with 8 GB unified memory.
- **Reason:** The project must remain usable and reproducible on the available hardware.
- **Consequence:** Data preparation must stream or use bounded chunks. Training must support Apple MPS and small batches. Large final experiments may use approved cloud compute, but cloud execution is not assumed for daily development.

## D-002: Separate development experiments from final training

- **Decision:** Use a small smoke dataset for pipeline checks, a bounded development dataset for model comparisons, and a larger dataset only after the pipeline is stable.
- **Reason:** Repeated full-dataset runs are not practical on an 8 GB laptop and make failures expensive.
- **Consequence:** Every experiment report must record dataset version, file count, split seed, model configuration, and hardware device.

## D-003: Split by source MIDI before windowing

- **Decision:** Train, validation, and test partitions are assigned by source MIDI file before overlapping sequence windows are created.
- **Reason:** Window-level splitting can place near-duplicate windows from one song in multiple partitions and inflate validation scores.
- **Consequence:** The split manifest becomes part of the evaluation artifact and must not be regenerated casually.

## D-004: Use next-token prediction

- **Decision:** The primary training objective predicts token `t+1` from tokens through `t` at every sequence position.
- **Reason:** The current eight-step target setup does not match generation, which uses only the final output position to predict one next token.
- **Consequence:** Training and inference share the same objective. A multi-step decoder is out of scope for this week.

## D-005: Preserve musical timing in the representation

- **Decision:** The target representation will be event-based and must represent note starts, note ends or durations, time shifts, velocity, and instrument information where available.
- **Reason:** Pitch-only tokens with fixed output durations cannot represent rhythm, rests, dynamics, or real polyphony.
- **Consequence:** Existing pitch-only generation is treated as a prototype baseline, not the final musical representation.

## D-006: Establish a simple baseline before attention

- **Decision:** Compare a statistical baseline and a compact unidirectional LSTM before evaluating the attention model.
- **Reason:** A more complex architecture is not evidence of improvement by itself.
- **Consequence:** Attention is accepted only if it improves agreed metrics under the same data, seed, and training budget.

## D-007: Do not use perplexity as the only quality claim

- **Decision:** Evaluation combines predictive, technical validity, musical-distribution, originality, latency, and human-review metrics.
- **Reason:** Lower next-token loss does not necessarily produce better-sounding music.
- **Consequence:** README and deployment claims must use exact report values and name the evaluation scope.

## D-008: Stream large data

- **Decision:** Keep training examples in bounded chunk files and load one chunk or bounded batches at a time.
- **Reason:** Merging tens of millions of windows into one tensor can exhaust 8 GB unified memory.
- **Consequence:** The existing merge-to-one-file path is not the default training path.

## D-009: MPS first, CPU fallback

- **Decision:** Training selects Apple MPS when available and falls back to CPU with an explicit log message.
- **Reason:** The M1 GPU is the available local accelerator, but MPS support is not identical to CUDA support.
- **Consequence:** CUDA-specific AMP, pinned-memory assumptions, and CUDA-only code are not accepted without device-specific verification.

## D-010: Deployment claims require verification

- **Decision:** A deployment is not considered complete because a frontend URL exists or a build succeeds.
- **Reason:** The actual model artifact, backend readiness, generation request, returned project output, and cold-start behavior must be checked.
- **Consequence:** Hosted verification must record URL, commit or artifact revision, request result, latency, and any known free-tier limitations.

## D-011: Choose the data modality by task fit

- **Decision:** MIDI remains the initial baseline, but the project will explicitly evaluate whether symbolic MIDI, extracted audio features, or raw MP3/audio generation best fits the intended product.
- **Reason:** MP3 is not automatically easier. Raw audio contains far more samples, requires substantially more storage and compute, and needs a different model and evaluation stack.
- **Consequence:** The first day includes a modality decision gate. Do not rewrite the full pipeline for MP3 until the task, dataset rights, compute budget, and evaluation method are defined.

## D-012: Model family is not fixed

- **Decision:** LSTM is a baseline, not a permanent architecture requirement.
- **Reason:** Compact Transformers, temporal convolutional models, GRU variants, and pretrained audio or symbolic music models may fit the task better.
- **Consequence:** Model comparisons must use the same data split, training budget, generation protocol, and evaluation report. A larger model is not automatically a better result on the available hardware.

## D-013: Google Colab is the approved scale-up path

- **Decision:** Google Colab may be used for larger training runs after the pipeline works locally.
- **Reason:** Colab can provide GPU memory and compute that the 8 GB M1 cannot provide reliably.
- **Consequence:** Colab notebooks must pin the repository commit, dataset version, configuration, random seed, dependencies, and output artifact location. Local development remains the source of truth for debugging and evaluation.

## D-014: First-week modality and model decision

- **Decision:** The first-week implementation target is symbolic MIDI generation using the existing compact unidirectional recurrent model as a baseline. MP3/audio remains an evaluated follow-up path, not a simultaneous rewrite.
- **Reason:** The current repository, seed data, API, and output path are MIDI-based. This gives the project a bounded, testable baseline on the M1 while leaving room to compare audio approaches in a separate experiment.
- **Consequence:** Do not implement a new audio representation or alternative model on Day 1. After the baseline and evaluation harness exist, compare a compact GRU or Transformer against the recurrent baseline under the same budget. The final architecture remains evidence-based.

## D-015: 250-song LMDClean pilot benchmark

- **Decision:** Before using the larger LMDClean corpus, build and evaluate a deterministic pilot subset of approximately 250 songs. Use the existing 10-song dataset only as a smoke test.
- **Reason:** A bounded pilot makes the full preprocessing, training, evaluation, and comparison loop affordable on the available hardware. It also exposes representation, leakage, memory, and reproducibility problems before a larger run.
- **Consequence:** The pilot is a development benchmark, not the final generalization claim. Select the pilot files with a recorded seed, target a 70/15/15 train/validation/test split, split by source song before windowing, and group by artist or album when reliable metadata is available. Record the exact post-grouping counts and freeze the pilot test split before model comparison.

## D-016: Controlled model comparison on one frozen pilot

- **Decision:** Compare the compact LSTM, GRU, and compact Transformer on the same pilot source split, token representation, sequence length, random seed policy, training budget, and evaluation sample count.
- **Reason:** Changing the dataset or training budget between models makes quality differences uninterpretable.
- **Consequence:** Every result must report predictive, MIDI-validity, musical-statistics, originality, latency, and resource metrics. A model is selected only after the comparison report records whether it improved, degraded, or did not conclusively change the agreed metrics.

## D-017: Larger-corpus scale-up gate

- **Decision:** Do not start larger LMDClean training until the 250-song pilot has a reproducible loader, stable bounded training run, frozen evaluation report, verified MIDI serialization, and documented resource profile.
- **Reason:** More data cannot repair a leaky split, broken target alignment, incomplete representation, or unstable training loop.
- **Consequence:** The larger run must pin the pilot-approved preprocessing configuration, repository revision, dataset version, split policy, seed, model configuration, and output artifact location. The final held-out evaluation must not be used for model tuning.
