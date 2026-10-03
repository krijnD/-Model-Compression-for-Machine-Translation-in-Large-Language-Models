# Rebuild every figure and table of the paper from results/ and check the hand-written tables (CPU only).
#   make paper            figures + appendix table + checks of Tables 1 and 2
#   make PY=../venv/bin/python paper
#   make paper && git diff --exit-code    rebuilt files are byte-identical to the committed ones
PY ?= python
export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH))
# fixed PDF timestamp, so rebuilt figures are byte-identical
export SOURCE_DATE_EPOCH := 1759449600

.PHONY: paper figures tables check
paper: figures tables check

figures:
	$(PY) scripts/paper/plot_rq1_heatmap.py
	$(PY) scripts/paper/plot_rq2_tradeoff.py
	$(PY) scripts/paper/plot_nll_w3.py
	$(PY) scripts/paper/plot_heatmap_adapter.py

tables:
	$(PY) scripts/paper/make_full_table.py

check:
	$(PY) scripts/paper/make_main_table.py
	$(PY) scripts/paper/make_failures_table.py
