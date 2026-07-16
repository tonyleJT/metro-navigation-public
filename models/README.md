# Model Weights

Place custom inference weights in `models/weights/`:

- `yolo11m.pt` — metro infrastructure detector;
- `segformer.pt` — three-class SegFormer checkpoint (`background`, `blindway`, `curb_ramp`);
- `yoloOCR.pt` — sign-oriented YOLO detector.

Large binaries should be stored through Git LFS or attached to a versioned release, not repeatedly committed to ordinary Git history. The repository `.gitattributes` already declares common weight formats for Git LFS.

Before publishing weights, add a model card containing:

- training-data provenance and permissions;
- class names and label IDs;
- training configuration and code commit;
- evaluation split and metrics;
- intended use and prohibited/high-risk use;
- known failure modes and station-specific assumptions;
- framework/version compatibility; and
- SHA-256 checksums.
