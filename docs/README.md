# Documentation

This directory contains the Typst manual and JSON examples.

- `manual.typ`: manual source.
- `qbp-sim.pdf`: rendered manual.
- `examples/basic_config.json`: minimal simulation config.
- `examples/matrix_config.json`: matrix config used by docs and smoke tests.

Build the manual:

```bash
typst compile docs/manual.typ docs/qbp-sim.pdf
```

Build Python artifacts:

```bash
uv build
```

The GitHub Actions `build` workflow runs tests, renders the manual, builds the wheel and source distribution, and uploads those files as workflow artifacts.
