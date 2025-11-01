# Contributing Guidelines

Thank you for your interest in contributing to Lattice Fourier Analyzer (LFA)!

## Getting Started
1. Fork the repository and clone your fork.
2. Create a virtual environment (Python 3.12 recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
4. Run the basic test/lint suite (see `/release/TEST_LINT_COMMANDS.csv`).

## Coding Standards
- Follow the existing layered architecture (see `context.md`).
- Run `black`, `ruff`, and `mypy` before opening a PR.
- Cover new logic with unit tests; use pytest-qt for GUI pieces.
- Keep UI strings/user-facing text in English.

## Contribution Workflow
1. Create a feature branch: `git checkout -b feature/my-improvement`
2. Commit changes with descriptive messages.
3. Push your branch and open a Pull Request describing:
   - Motivation / summary
   - Implementation details
   - Testing performed
4. Link to relevant issues or discussion threads.
5. Respond to code review feedback promptly.

## Citation / Publication
- Use `release/RELEASE_NOTES_TEMPLATE.md` to capture highlights.
- Archive tagged releases on Zenodo to obtain a DOI.
- Include citation text in the README once DOI is available.

## Reporting Issues
- Use the issue tracker with a clear title and reproduction steps.
- Attach sample data if applicable (sanitized / permitted for sharing).

## Code of Conduct
- Be respectful and collaborative.
- Focus discussions on the code and project priorities.
