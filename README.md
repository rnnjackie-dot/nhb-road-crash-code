# NHB road-crash analysis code

Public code accompanying the NHB submission **A knowledge-grounded language-model agent uncovers psychological mechanisms of road crashes from police interrogation records**.

## Repository contents

- `code/pipeline/`: OCR batching, extraction, knowledge-graph construction, hypothesis generation, and automated scoring scripts.
- `code/generate_figures.py` and `code/generate_figures_extended.py`: figure and statistics generation scripts.
- `requirements.txt`: Python dependencies for the released scripts.

The source records, OCR inputs, instantiated knowledge graph, hypothesis files, scored results, expert ratings, figures, and other derived data are intentionally excluded because they are governed by the originating data-use agreement and must not be publicly distributed.

## Running the restricted pipeline

The pipeline scripts require lawful access to the source records and local OCR tooling. Configure all paths and credentials through environment variables; never commit credentials or source records.

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:THEORY_DOCX = "path/to/theory.docx"
$env:OCR_DIR = "path/to/ocr_results"
$env:OUTPUT_DIR = "path/to/output"
$env:HYPOTHESIS_DIR = "path/to/hypotheses"
$env:HYP_DIR = "path/to/hypotheses"
```

The code expects the corresponding private data package or an approved controlled-access copy. Public users cannot reproduce the data-dependent analyses from this repository alone.

## Reuse

No open-source license has been selected yet. Copyright remains with the authors and applicable data custodians. Please contact the corresponding author before reuse or redistribution.
