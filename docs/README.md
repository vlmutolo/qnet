# qbp-sim docs

This directory contains the user-facing Typst manual and small JSON examples.

- `manual.typ`: canonical documentation source.
- `qbp-sim.pdf`: rendered manual.
- `examples/basic_config.json`: minimal simulation config.
- `examples/matrix_config.json`: tiny experiment matrix used in the manual and smoke tests.

Build the PDF with:

```bash
typst compile docs/manual.typ docs/qbp-sim.pdf
```

Build Python package artifacts with:

```bash
uv build
```

The GitHub Actions `build` workflow runs tests, renders the manual, builds the wheel/source
distribution, and uploads those files as workflow artifacts.
