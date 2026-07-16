# Code Review and Publication Report

## Scope

This review covered the complete legacy source snapshot supplied with the project:

- 9 Python files, approximately 2,391 source lines;
- YOLO metro-object detection;
- YOLO sign-oriented detection;
- SegFormer-B0 path segmentation;
- phase-aware fusion and speech guidance;
- the online/video runner and benchmark; and
- five model/export files totaling approximately 168 MB.

The cleaned repository is deliberately a **conservative refactor**. It preserves the project’s route-specific classes, confidence thresholds, safe-region scoring, look-ahead geometry, and intended route phases unless an issue was clearly defective. No claim is made that unvalidated behavior changes are correct merely because they are cleaner.

## High-priority defects found in the legacy snapshot

### Import and configuration failures

- The source expected a package named `MetroPaper`, but the snapshot did not provide a valid installable package structure.
- `models/ss_model.py` imported `MetroReport`, which appears to be a package-name typo.
- `main.py` and the benchmark modified `sys.path` at runtime.
- Model and video paths were hard-coded to one Windows `T:` drive.

**Resolution:** an installable `src/metro_navigation` package, `pyproject.toml`, typed configuration dataclasses, relative defaults, command-line overrides, and environment-variable overrides.

### Detection information was discarded

When an escalator entry was detected, the legacy object wrapper returned only escalator detections and removed every other valid object from `detections`. This made downstream fusion blind to simultaneous gates, stairs, ticket booths, and other context.

**Resolution:** retain all accepted detections while also exposing a dedicated, confidence-sorted escalator subset.

### Segmentation could remain disabled

The fusion logic disabled SegFormer whenever an escalator was visible. The transition code that should restore normal behavior was commented out or depended on later route-specific evidence, so segmentation could remain disabled after the escalator disappeared.

**Resolution:** segmentation is re-enabled immediately after escalator-only override ends. A unit test protects this behavior.

### Route phase could move backward

Scene inference was allowed to replace the current phase after a confirmation count, including with an earlier phase. Missed detections or video cuts could therefore regress navigation state.

**Resolution:** inferred transitions are forward-only. Explicit route resets should be a separate operation rather than accidental inference.

### SEARCHING fallback did not match the stated design

In the entry phase, the legacy code provided no local segmentation guidance when a stair landmark was absent. This conflicts with the intended fallback in which global landmarks guide first when available, while safe-region segmentation remains available when no reliable landmark is seen.

**Resolution:** entry guidance now falls back to the segmentation safe target.

### Speech and display state were unstable

- More than one forced message could be queued during one frame.
- Later messages could overwrite the user-interface text generated earlier in the same frame.
- An unchanged direction was never repeated, which can leave a user without reinforcement during a long movement.
- Cooldowns used wall-clock time, which can jump if the system clock changes.

**Resolution:** one accepted message per update, explicit event priority, controlled repetition of unchanged guidance, and monotonic timing.

### Incorrect semantic instruction

A `stair node` could trigger the sentence “Get on elevator.”

**Resolution:** it now announces stairs rather than inventing an elevator.

### Hard-coded station announcement

The legacy entry phase announced “Tan Cang station” after seeing a stair, even though a stair does not establish station identity.

**Resolution:** station announcement is disabled by default and can be supplied explicitly through `--station-name` for a known route. It is not inferred from a generic stair class.

### Stale segmentation visualization

When segmentation was skipped for frame-rate control, the cached blended overlay belonged to an earlier camera frame. Reusing it visually aligned old image content with a new frame.

**Resolution:** segmentation stores a class-color mask rather than a previously blended frame. The mask is composited onto the current frame during rendering.

### Benchmark memory behavior

The legacy component benchmark retained a whole video in memory. Long or high-resolution videos could consume several gigabytes.

**Resolution:** each benchmark pass streams frames from a reopened capture. Only the lightweight fusion-input cache remains for the fusion-only measurement.

## Repository engineering changes

- Standard `pyproject.toml` package metadata and console command: `metro-nav`.
- `src/` layout and explicit package modules.
- Ruff formatting/linting configuration.
- Mypy configuration and full-source type check.
- Pytest unit tests for clock geometry, segmentation mode hysteresis, phase monotonicity, fallback guidance, speech priority, guidance repetition, and escalator recovery.
- GitHub Actions CI for formatting, linting, and tests.
- `.gitignore` for caches, build outputs, videos, datasets, and runtime results.
- `.gitattributes` rules for Git LFS model formats and normalized line endings.
- Research-prototype safety warning, model documentation, contribution rules, citation metadata, and reproducibility checklist.
- Compatibility launchers retained as `main.py` and `benchmark_components.py`.
- Legacy TensorRT exports omitted because the reviewed Python runtime does not load them.

## Intentionally not changed without evidence

The following are empirical design choices, not formatting issues. Changing them without route videos, annotations, and regression metrics would be guesswork:

- YOLO confidence thresholds and class-specific thresholds;
- SegFormer morphology size and minimum connected-region area;
- safe-zone forward/lateral utility weights;
- look-ahead corridor ratios, target depth, minimum row support, and EMA coefficient;
- turn timing and route-specific left/right assumptions;
- sign pairing classes and spatial thresholds; and
- phase meanings and station-specific class names.

These values are now named and centralized so they can be tuned from evidence rather than edited throughout the implementation.

## Validation completed

The cleaned source passed:

```text
Ruff format check: passed
Ruff lint: passed
Pytest: 12 passed
Mypy: no issues in 16 source files
Python byte-code compilation: passed
Package sdist/wheel build: passed
CITATION.cff schema validation: passed
Vulture unused-code scan at 80% confidence: no findings
```

The average cyclomatic complexity reported by Radon is grade A. Two route-heavy functions and the main processing loop remain relatively complex; splitting them further should wait until end-to-end regression fixtures exist, because mechanical decomposition of safety-relevant state logic can still introduce ordering errors.

## Validation not completed

Actual YOLO/SegFormer inference was not executed in the review environment because the environment did not contain the complete GPU/ML runtime and no approved regression video was supplied. Therefore:

- checkpoint compatibility with the selected dependency versions is not yet proven;
- frame-by-frame behavior has not been compared with the conference demonstration;
- TensorRT/CUDA performance has not been remeasured;
- route completion and instruction correctness have not been validated; and
- no human-subject or safety validation is implied.

Before replacing the public branch, run both old and cleaned versions on the same privacy-safe videos and compare detections, masks, targets, phase transitions, speech events, and timing logs.

## Publication blockers and decisions

1. **License approval:** confirm ownership and approval with the coauthor, institution, and grant administrator. Review Ultralytics and SegFormer upstream terms before selecting the repository license or redistributing derived weights.
2. **Model release:** use Git LFS or a versioned release asset; publish checksums and a model card. The supplied checkpoint hashes are in `models/SHA256SUMS.txt`.
3. **Data/privacy:** do not publish raw metro footage or labels until consent, face/privacy, operator permission, and institutional requirements are resolved.
4. **Reproducibility:** the current snapshot is inference-only. Add training scripts, split manifests, evaluation scripts, seeds, dependency lock information, and checkpoint-to-experiment mapping before calling it reproducible.
5. **Safety wording:** describe it as a research prototype, not a deployable or certified mobility aid. Detection mAP, mIoU, and pixel accuracy are not end-to-end navigation-safety measures.

## Safe GitHub migration sequence

Do not overwrite the existing branch immediately. Preserve history and review the change as a branch:

```bash
git checkout -b refactor/public-release
# Copy the cleaned repository contents into the working tree.
git lfs install
git add .
git commit -m "Refactor project for reproducible public release"
git push -u origin refactor/public-release
```

Then run a regression test, inspect the pull-request diff, and merge only after the paper behavior is confirmed. A repository name such as `phase-aware-metro-navigation` communicates the project more clearly than `thesis-draft`, but renaming is optional and should be done after links/citations are checked.
