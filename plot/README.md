# Plot Scripts

Use `run_all_plots.bat` to regenerate the current paper figures. The active scripts are the `*_mpl.py` files, which use `matplotlib`/`pandas` and write PNG files to `../Draft/Experiments-Result/plots`.

The older SVG scripts without the `_mpl` suffix are legacy drafts kept for traceability. They are not called by `run_all_plots.bat` and should not be used for the current LaTeX experiment section.

`plot_overall_bars_mpl.py` reads `bootstrap_overall_ci.csv` because that file was generated from the plain-prompt raw scores and contains both the plain overall point estimates and bootstrap confidence intervals.
