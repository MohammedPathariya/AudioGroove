# AudioGroove MLOps and Modality Roadmap

Updated: 2026-08-05

## Purpose

AudioGroove is being developed as an end-to-end MLOps system for symbolic
music generation. The immediate objective is not to claim a production music
generator. It is to establish a traceable chain from versioned data to
parallel HPC experiments, model selection, registry, deployment, and
monitoring.

The immediate modality is symbolic MIDI. Raw MP3/audio modeling is deferred
until the project has licensed recordings, a suitable representation, enough
data, and an audio-specific evaluation protocol.

## Lifecycle

```text
Versioned corpus
  -> frozen 250-song pilot
  -> deterministic preprocessing
  -> Slurm GPU experiments
  -> MLflow comparison
  -> validation-based selection
  -> held-out test evaluation
  -> model registry
  -> API and frontend deployment
  -> monitoring and feedback
```

## Data and reproducibility contract

Every training run must identify:

- source corpus and dataset revision
- selected-song manifest and content hashes
- source-level split manifest
- vocabulary and representation revision
- sequence length and chunk configuration
- preprocessing code commit
- model code commit
- random seeds
- HPC environment and GPU

The current pilot contains 250 songs split into 175 train, 37 validation, and
38 test songs. It has 190 distinct artists and no reliable genre metadata. It
is therefore a heterogeneous artist-disjoint symbolic benchmark, not a
genre-stratified benchmark.

The retained audit records source revision `cb79c82...`; the previous Colab
prepared-data record reported `bf670db4...`. These revisions must be reconciled
before HPC preprocessing is accepted as the same experiment.

## Experiment design

### Baseline comparison

Run one compact LSTM, one compact GRU, and one compact causal Transformer under
the same dataset, split, sequence length, effective batch size, training
budget, seed policy, and evaluation protocol.

### Configuration sweep

Use a deliberately small sweep rather than an unrestricted grid:

- three model sizes per family
- fixed five-epoch initial budget
- validation-only configuration selection
- at least three seeds for finalists
- one final held-out test evaluation

The comparison must report both quality and cost. A lower validation loss is
not enough to justify promotion if the model is unstable, invalid at
generation time, or disproportionately expensive.

## HPC execution

Big Red 200 is the approved training environment.

- Slurm project: `r00284`
- GPU partition for smoke tests: `gpu-debug`
- Production experiment partition: `gpu`, subject to allocation policy
- Python module: `python/gpu/3.11.5`
- Verified GPU: NVIDIA A100-SXM4 40 GB
- Personal scratch: `/N/scratch/mjpathar`

The first comparison uses independent jobs, not distributed training:

- one LSTM job on one GPU node
- one GRU job on one GPU node
- one Transformer job on one GPU node

This provides parallel experiment execution while keeping each run isolated
and easy to reproduce.

## MLflow design

Use one experiment such as:

`AudioGroove-250-Pilot-HPC`

Each model/configuration/seed is a separate run. A parent batch run may group
the baseline, sweep, or finalist jobs.

Log parameters including:

- model family and full architecture configuration
- parameter count
- dataset and representation revisions
- split and training seeds
- batch, accumulation, and sequence settings
- optimizer, learning rate, scheduler, and clipping
- Git commit
- Slurm job ID, node, GPU, and environment versions

Log metrics including:

- train and validation loss
- perplexity and token accuracy
- epoch duration and throughput
- peak GPU memory
- generation latency
- MIDI validity and failure rates

Log artifacts including:

- best and final checkpoints
- configuration JSON
- environment manifest
- Slurm stdout and stderr
- generated MIDI files
- evaluation JSON and CSV reports
- model compatibility metadata

The local file-store MLflow setup must not be assumed safe for concurrent HPC
writes. The HPC implementation must choose either a tracking server reachable
from compute nodes or isolated per-job stores with a verified consolidation
step.

## Evaluation and promotion

Validation data controls model and configuration selection. The 38-song test
split remains untouched until finalists are fixed.

Promotion requires:

1. successful checkpoint reload
2. compatible vocabulary and representation metadata
3. reproducible held-out evaluation
4. valid MIDI generation
5. documented resource profile
6. complete MLflow run and artifacts

After promotion, the API and frontend must be tested with the registered
artifact rather than an arbitrary local checkpoint.

## Genre and audio extensions

The current pilot should not be retroactively treated as a genre benchmark.
Future genre studies should create separate manifests with reliable labels and
separate train/validation/test splits.

The audio extension should begin with a feasibility study using owned or
licensed recordings. A practical order is:

1. rendered audio previews from generated MIDI
2. pretrained audio representation or neural codec experiments
3. short-segment audio continuation
4. larger audio model comparison only after data and evaluation gates pass

Raw waveform generation from a small collection of downloaded MP3s is not the
planned next step.
