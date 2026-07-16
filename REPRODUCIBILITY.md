# Reproducibility Checklist

The current repository is an inference release, not a complete reproduction package. To make the paper results independently reproducible, add the following items without exposing private or restricted data.

## Required research artifacts

1. Training scripts for both YOLO detectors and SegFormer-B0.
2. Exact dependency lock file and CUDA/cuDNN versions.
3. Dataset version identifiers and immutable train/validation/test split manifests.
4. Class-name files and annotation conversion scripts.
5. Random seeds and deterministic/non-deterministic settings.
6. Full hyperparameter configuration for each reported experiment.
7. Evaluation scripts that generate every metric in the paper.
8. Benchmark hardware details, power mode, warm-up protocol, and raw timing logs.
9. Checkpoint hashes and a mapping from each checkpoint to the corresponding experiment.
10. End-to-end test videos or privacy-safe derived fixtures that exercise each route phase.

## Important methodological gap

High segmentation and detection metrics do not by themselves validate navigation safety. The release should clearly separate:

- perception metrics;
- state-transition accuracy;
- instruction correctness;
- temporal stability and latency;
- route-completion outcomes; and
- user-safety outcomes.

A future evaluation should report false guidance events, missed warnings, phase-transition errors, recovery behavior, tail latency, and performance under crowd/occlusion conditions rather than relying only on mean model metrics.
