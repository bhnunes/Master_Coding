# DIAGSET Manuscript Source Bundle

This folder is the standalone source bundle for the DIAGSET manuscript.
It is intended to compile without relying on `reports/` or `latex_template/`
from the original repository.

## Contents

- `main.tex`: manuscript entrypoint.
- `references.bib`: manuscript bibliography database.
- `figures/`: figures referenced by `main.tex`.
- `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`: Elsevier CAS files needed by the manuscript.
- `thumbnails/`: Elsevier CAS icons loaded by `cas-common.sty` during `\maketitle`.
- `figs/`: Elsevier CAS sample assets retained with the copied template files.
- `support/`: manuscript support notes, including the no-rerun evidence index.

Existing PDF files in this folder are cited-paper copies and are not required
to compile the manuscript.

## Compile

From this folder:

```powershell
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

On Overleaf, set `main.tex` as the main document and use the XeLaTeX compiler.
