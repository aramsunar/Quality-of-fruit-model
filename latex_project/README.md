# LaTeX Project: Fruit Identification and Classification

This LaTeX project contains the complete documentation for the Group Mini-Project on fruit identification and classification using convolutional neural networks.

## Project Structure

```
latex_project/
├── main.tex                 # Main document file (compile this)
├── titlepage.tex           # Custom title page with NWU branding
├── chapters/               # Individual chapter files
│   ├── 01_introduction.tex
│   ├── 02_cnn_methodology.tex
│   ├── 03_part_a.tex
│   ├── 04_scenario1.tex
│   ├── 05_scenario2.tex
│   ├── 06_scenario3.tex
│   ├── 07_part_b.tex
│   └── 08_conclusion.tex
├── figures/                # Directory for figure files
└── tables/                 # Directory for table files
```

## Document Contents

### Front Matter
- Title page with NWU branding
- Table of Contents
- List of Tables
- List of Figures

### Main Content
1. **Introduction** - Project motivation, objectives, and methodology overview
2. **Convolutional Neural Network** - Dataset description, architecture details, and configuration parameters
3. **Part A: Single-Task Learning**
   - Scenario 1: Baseline RGB images
   - Scenario 2: Grayscale images
   - Scenario 3: Augmented images
4. **Part B: Multi-Task Learning** - Overview and reference to detailed analysis
5. **Conclusion** - Key findings, implications, and future directions

## Compilation Instructions

### Prerequisites
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (automatically installed with most distributions):
  - geometry
  - graphicx
  - booktabs
  - longtable
  - hyperref
  - xcolor
  - titlesec
  - fancyhdr
  - amsmath
  - caption
  - subcaption
  - enumitem

### Compiling the Document

#### Method 1: Command Line (Recommended)
```bash
cd latex_project
pdflatex main.tex
pdflatex main.tex  # Run twice for cross-references
```

#### Method 2: LaTeX Editor
Open `main.tex` in your preferred LaTeX editor (TeXstudio, Overleaf, TeXmaker, etc.) and compile using the editor's build function.

#### Method 3: Automated Build
```bash
cd latex_project
latexmk -pdf main.tex
```

### Output
The compiled PDF will be named `main.pdf` in the `latex_project/` directory.

## Customization

### Adding Figures
1. Place figure files (.png, .pdf, .jpg) in the `figures/` directory
2. Reference them in the document using:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/your_figure.png}
\caption{Your caption here}
\label{fig:your_label}
\end{figure}
```

### Modifying Colors
The NWU purple color is defined in `main.tex`:
```latex
\definecolor{nwupurple}{RGB}{128,0,128}
```

### Adjusting Page Layout
Page geometry settings are in `main.tex`:
```latex
\geometry{
    a4paper,
    left=25mm,
    right=25mm,
    top=25mm,
    bottom=25mm
}
```

## Document Information

- **Authors**:
  - Angelina Ramsunar - 41081269
  - Stefan Du Plooy - 40954129
  - Rikus Swart - 42320755
- **Institution**: North-West University
- **Module Code**: ITRI 626
- **Date**: 12-11-2025

## Notes

### Part B Reference
Part B (Multi-Task Learning) is referenced in the document but contains only an overview. Detailed results and analysis for Part B are available in the separate document `PART_B.md` in the parent directory.

### Figure Placeholders
The current document includes references to figures but does not include the actual figure files. To include figures:
1. Export figures from the Python analysis scripts
2. Place them in the `figures/` directory
3. Uncomment or add `\includegraphics` commands in the appropriate chapter files

### Table Formatting
All tables use the `booktabs` package for professional-quality formatting. Use:
- `\toprule` for top horizontal line
- `\midrule` for middle horizontal lines
- `\bottomrule` for bottom horizontal line
- Avoid vertical lines for cleaner appearance

## Troubleshooting

### Common Issues

**Issue**: "File not found" errors
- **Solution**: Ensure all chapter files exist in the `chapters/` directory
- **Solution**: Check that file paths in `\include{}` commands match actual filenames

**Issue**: Cross-references showing "??"
- **Solution**: Run pdflatex twice (first run creates .aux file, second resolves references)

**Issue**: Missing packages
- **Solution**: Install missing packages using your LaTeX distribution's package manager
  - TeX Live: `tlmgr install <package-name>`
  - MiKTeX: Packages install automatically on first use

**Issue**: Table of contents not updating
- **Solution**: Delete auxiliary files (.aux, .toc, .lof, .lot) and recompile

### Clean Build
To remove auxiliary files and start fresh:
```bash
cd latex_project
rm -f *.aux *.log *.toc *.lof *.lot *.out *.synctex.gz
pdflatex main.tex
pdflatex main.tex
```

## Version Control

When using Git, consider adding the following to `.gitignore`:
```
*.aux
*.log
*.toc
*.lof
*.lot
*.out
*.synctex.gz
*.fdb_latexmk
*.fls
*.nav
*.snm
*.vrb
```

## Contact

For questions or issues related to this LaTeX project, contact the authors through the North-West University.

## License

This document is submitted as part of academic coursework for ITRI 626 at North-West University.
