#!/usr/bin/env bash
# Compila documentation/paper.tex -> documentation/paper.pdf
#
# TeX e' installato a livello UTENTE in ~/.TinyTeX (TeX Live 2026 via TinyTeX),
# binari in ~/.local/bin: niente sudo, niente pacchetti di sistema, e su g2 non
# tocca nulla degli altri utenti. Reinstallabile con:
#   curl -sSL https://yihui.org/tinytex/install-bin-unix.sh | sh
#   ~/.local/bin/tlmgr install ieeetran cite algorithms psnfss courier times helvetic
#
# Gli ausiliari (.aux .bbl .log .out) restano in documentation/.build/, cosi'
# `documentation/` non si riempie di scarti; esce solo il PDF.
#
#   bash scripts/build_paper.sh            # 3 passate + bibtex
#   bash scripts/build_paper.sh --quick    # una passata sola, per un check rapido
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$REPO/documentation"
BUILD="$DOC/.build"
JOB="paper"

export PATH="$HOME/.local/bin:$PATH"
command -v pdflatex >/dev/null || {
  echo "pdflatex non trovato. Installa TinyTeX (vedi in testa a questo script)." >&2
  exit 1
}

mkdir -p "$BUILD"
cd "$DOC"                       # le figure sono incluse per path relativo

run_tex() {
  pdflatex -output-directory="$BUILD" -interaction=nonstopmode "$JOB.tex" >/dev/null 2>&1 || true
}

run_tex
if [[ "${1:-}" != "--quick" ]]; then
  # i due path servono con -output-directory: bibtex gira dentro $BUILD e da li'
  # non vedrebbe references.bib ne' IEEEtran.bst. I due punti finali dicono
  # "poi cerca anche dove cercheresti normalmente".
  (cd "$BUILD" && BIBINPUTS="$DOC:" BSTINPUTS="$DOC:" bibtex "$JOB" >/dev/null 2>&1) || true
  run_tex
  run_tex
fi

cp "$BUILD/$JOB.pdf" "$DOC/$JOB.pdf"

echo "-> $DOC/$JOB.pdf  ($(du -h "$DOC/$JOB.pdf" | cut -f1))"
# `grep -c` esce 1 quando non trova nulla, che sotto `set -e` e' proprio il caso
# buono: ogni conteggio va protetto con `|| true`.
echo "errori: $(grep -c '^!' "$BUILD/$JOB.log" || true)"
for w in 'Reference' 'Citation' 'Label(s) may have changed'; do
  n=$(grep -c "LaTeX Warning: $w" "$BUILD/$JOB.log" || true)
  [[ "$n" != "0" ]] && echo "avvisi \"$w\": $n"
done
echo "scatole over/underfull: $(grep -cE '^(Overfull|Underfull)' "$BUILD/$JOB.log" || true)"
exit 0
