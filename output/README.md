# Outputs

`reference/` contains a checked execution of `workflow/reproduce.py` from the
reference environment. `reproduced/` is created by `make reproduce`. Compare
the numeric CSV/JSON content and the generated manifest; PDF byte hashes can
vary across Matplotlib, font, and PDF-backend versions even when plotted values
are identical.
