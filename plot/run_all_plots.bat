@echo off
setlocal
cd /d "%~dp0\.."
set UV_CACHE_DIR=.uv-cache

if exist ".venv\Scripts\python.exe" (
  set PYTHON=.venv\Scripts\python.exe
) else (
  set PYTHON=uv run --python C:\Python313\python.exe python
)

%PYTHON% plot\plot_overall_bars_mpl.py
%PYTHON% plot\plot_subset_family_radar_mpl.py
%PYTHON% plot\plot_subset_radar_23_mpl.py
%PYTHON% plot\plot_finite_sample_representative_mpl.py
%PYTHON% plot\plot_finite_sample_all_models_mpl.py
%PYTHON% plot\plot_experiment3_bootstrap_summary_mpl.py
%PYTHON% plot\plot_chattemplate_overall_bars_mpl.py
%PYTHON% plot\plot_chattemplate_subset_family_radar_mpl.py
%PYTHON% plot\plot_chattemplate_subset_radar_23_mpl.py

endlocal
