.PHONY: all install verify-data test reproduce smoke full full-obm checksums archive

PYTHON ?= python3
PYTHONPATH := src

all: verify-data test reproduce

install:
	$(PYTHON) -m pip install -r requirements-reproducibility.txt
	$(PYTHON) -m pip install -e .

verify-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/regenerate_data.py

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

reproduce:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/reproduce.py --output-dir output/reproduced

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/run_simulations.py --mode smoke --output output/simulation_smoke.jsonl
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/run_obm.py --mode smoke --output-dir output/obm_smoke

full:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/run_simulations.py --mode full --output output/full_simulation_records.jsonl

full-obm:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) workflow/run_obm.py --mode full --output-dir output/obm_full

checksums:
	$(PYTHON) workflow/make_checksums.py

archive: checksums
	$(PYTHON) workflow/build_archive.py
