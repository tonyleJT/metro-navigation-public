# Licensing Decision Required

No final project license is included yet. Publishing a public repository without a license does **not** make it open source; normal copyright restrictions remain in effect.

Before selecting a license, confirm:

1. who owns the code under the university grant and thesis/conference arrangements;
2. whether the coauthor and institution approve public redistribution;
3. whether the custom datasets and model weights may be redistributed;
4. whether station imagery contains privacy, operator, or location restrictions; and
5. how the chosen license interacts with Ultralytics YOLO licensing.

Ultralytics currently offers AGPL-3.0 and enterprise licensing options. A permissive license placed on this repository does not remove obligations that may arise from distributing or deploying an application built with AGPL-licensed Ultralytics software. For an academic open-source release, AGPL-3.0 may be the simplest compatibility path, but this is a legal/project-governance decision rather than a formatting choice.

The SegFormer backbone model card currently labels its license as “other,” so its upstream terms and the ADE20K-related provenance should also be checked before redistributing derived weights.

After approval, add a standard `LICENSE` file and update `pyproject.toml`, the README, model card, and release notes consistently.
