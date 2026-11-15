$pdf_mode = 1;
$out_dir = 'build';
@default_files = ('main.tex');
# Run bibtex from this folder: MiKTeX's bibtex can't read latexmk's Unix-style BIBINPUTS under Git Bash.
$bibtex_fudge = 0;
