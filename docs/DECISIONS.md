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

## D-013: Big Red 200 is the approved scale-up path

- **Decision:** Big Red 200 is the primary scale-up environment for the 250-song pilot and subsequent controlled experiments.
- **Reason:** The account has verified Slurm GPU access through project `r00284`, with an A100 GPU and CUDA-enabled PyTorch. This provides reproducible batch execution without relying on a temporary Colab runtime.
- **Consequence:** HPC jobs must pin the repository commit, dataset version, configuration, random seed, dependencies, Slurm project, GPU request, and artifact locations. Colab is not part of the current training path.

## D-014: MIDI-first MLOps benchmark

- **Decision:** Complete the symbolic MIDI MLOps benchmark before beginning raw MP3/audio modeling. Compare a compact LSTM, GRU, and causal Transformer on the frozen pilot, then connect the selected artifact to the API and frontend.
- **Reason:** MIDI is interpretable, bounded, and already supported by the repository. Raw audio requires a separate licensed dataset, representation, model family, and evaluation stack.
- **Consequence:** Rendered audio previews may improve the product experience, but raw audio training is deferred until its data and evaluation gates are approved.

## D-015: 250-song LMDClean pilot benchmark

- **Decision:** Before using the larger LMDClean corpus, build and evaluate a deterministic pilot subset of approximately 250 songs. Use the existing 10-song dataset only as a smoke test.
- **Reason:** A bounded pilot makes the full preprocessing, training, evaluation, and comparison loop affordable on the available hardware. It also exposes representation, leakage, memory, and reproducibility problems before a larger run.
- **Consequence:** The pilot is a development benchmark, not the final generalization claim. Select the pilot files with a recorded seed, target a 70/15/15 train/validation/test split, split by source song before windowing, and group by artist or album when reliable metadata is available. Record the exact post-grouping counts and freeze the pilot test split before model comparison.

## D-016: Controlled model comparison on one frozen pilot

- **Decision:** Compare the compact LSTM, GRU, and compact Transformer on the same pilot source split, token representation, sequence length, random seed policy, training budget, and evaluation sample count. Use Dask for bounded preprocessing and MLflow for local, machine-readable experiment tracking.
- **Reason:** Changing the dataset or training budget between models makes quality differences uninterpretable.
- **Consequence:** Every result must report predictive, MIDI-validity, musical-statistics, originality, latency, and resource metrics. Dask task configuration, MLflow run IDs, Slurm job IDs, and GPU metadata must be recorded with the dataset revision and Git commit. Validation results select configurations; the frozen test split is evaluated only after selection.

## D-018: Dask and MLflow experiment infrastructure

- **Decision:** Use Dask to parallelize MIDI parsing and bounded feature or chunk preparation, and use MLflow to track pilot training and benchmark runs. HPC tracking must use a concurrency-safe tracking server or isolated per-job stores with verified consolidation.
- **Reason:** The project needs a traceable preprocessing and training history that can support claims about dataset scale, runtime, experiments, and model selection without relying on undocumented terminal output.
- **Consequence:** Dask work must preserve source-level split assignments and deterministic ordering. MLflow must log dataset revision, source and window counts, selection and split seeds, model configuration, device, Slurm metadata, training budget, metrics, resource measurements, checkpoints, manifests, and benchmark reports. Dask and MLflow integration is not considered verified until parallel HPC runs complete and their artifacts can be reopened.

## D-019: Genre and audio are separate follow-up tracks

- **Decision:** Treat the current 250-song pilot as a heterogeneous artist-disjoint symbolic benchmark, not a genre-stratified benchmark. Create separate genre manifests and audio experiments rather than relabeling or replacing the current pilot.
- **Reason:** The corpus layout has no reliable genre metadata, and raw MP3 training introduces substantially different rights, data, representation, compute, and evaluation requirements.
- **Consequence:** The current model-selection decision applies only to the symbolic MIDI pilot. Genre-specific or audio conclusions require their own frozen datasets and MLflow experiments.

## D-017: Larger-corpus scale-up gate

- **Decision:** Do not start larger LMDClean training until the 250-song pilot has a reproducible loader, stable bounded training run, frozen evaluation report, verified MIDI serialization, and documented resource profile.
- **Reason:** More data cannot repair a leaky split, broken target alignment, incomplete representation, or unstable training loop.
- **Consequence:** The larger run must pin the pilot-approved preprocessing configuration, repository revision, dataset version, split policy, seed, model configuration, and output artifact location. The final held-out evaluation must not be used for model tuning.

## D-020: Shared pilot model and training contract

- **Decision:** Build the LSTM, GRU, and causal Transformer behind one
  next-token interface and train them through one configuration-driven runner.
  Freeze small, baseline, large, and larger profiles in
  `training/configs/pilot_experiments.json`. Use five full epochs, batch size
  64, AdamW, weight decay `1e-4`, gradient clip 1.0, validation-based early
  stopping, and CUDA AMP for the initial profiles. Use learning rate `1e-3`
  for LSTM/GRU and `3e-4` for Transformer.
- **Reason:** Separate trainers or undocumented command-line changes would
  confound architecture differences with data order, optimizer behavior,
  metrics, checkpoint semantics, and generation settings.
- **Consequence:** Every run streams the same deterministic chunk order for its
  seed, logs parameter count and full configuration, validates checkpoint data
  and vocabulary compatibility, and generates from a validation seed. Test
  evaluation is absent from the training command. A separate evaluator requires
  a frozen validation-only selection manifest after finalist selection. The
  baseline configurations are not parameter matched; quality must be
  interpreted alongside parameter count, runtime, and memory.

## D-021: Fit symbolic vocabulary on training data only

- **Decision:** Build the pilot vocabulary from the 175 training songs only.
  Map unseen validation and test tokens to `<UNK>`, record split-level OOV
  counts and rates, and classify revision `bf670db4...` and its completed runs
  as `preliminary-v1`.
- **Reason:** Building token identities from validation and test songs leaks
  held-out distribution information into a fitted preprocessing artifact.
- **Consequence:** The corrected revision is `a68aee4e...` with vocabulary size
  18,849. All model selection must be repeated on that revision. The final test
  requires a frozen validation-only selection manifest and an atomic one-time
  evaluation gate.

## D-022: Use GRU-small for Render Free deployment

- **Decision:** Use the recovered 250-song GRU-small artifact, rather than the
  research-selected 2,500-song GRU-large artifact, for the Render Free
  deployment candidate.
- **Reason:** The GRU-large worker exceeded the free-tier memory envelope
  during generation. The GRU-small inference-only package passed a local
  512 MB constrained-container gate with 236.2 MiB peak memory and no cgroup
  allocation denials or OOM events.
- **Consequence:** The deployment package is `gru_small_250`, with a matching
  18,849-token vocabulary, `deploy.pt`, and deployment manifest. It uses a
  CPU-only Torch build. Hosted Render verification remains required before any
  public deployment claim.
