# Contributing

1. Create a focused branch and keep behavior changes separate from formatting-only changes.
2. Run `ruff format .`, `ruff check .`, and `pytest` before opening a pull request.
3. Add tests for state-machine changes, especially phase transitions and speech priority.
4. Do not commit private station videos, personal data, credentials, or unapproved model weights.
5. Document any parameter change with the validation evidence that supports it.
6. Treat safety-related behavior as a reviewed design change, not a cosmetic refactor.
