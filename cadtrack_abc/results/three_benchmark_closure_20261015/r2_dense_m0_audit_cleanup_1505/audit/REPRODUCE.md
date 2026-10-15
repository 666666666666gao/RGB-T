Run only the audit checkers, never the sealed project runners. All paths in this directory are local/private.

`E:/python.exe -B check_artifacts.py`
`E:/python.exe -B check_source_stdlib.py`
`E:/python.exe -B check_orchestration.py`
`E:/python.exe -B check_lineage_and_coverage.py`
`E:/python.exe -B audit_io.py seal --phase after`

For exact stdout/stderr/exit capture, use `run_check.py <new_unique_tag> <checker.py> [arguments]` from this directory. This writes only inside this audit directory. Use a new tag because old attempts are immutable. The NumPy checker is retained as a failed attempt, not required to reproduce the successful stdlib audit. No Torch/model imports or installs are needed.

Input paths pass through a Windows junction to `D:/Program Files/UserCache/gb/codex_tmp`; this explains canonical paths in subprocess receipts. It is the same authorized output directory.

The private JSON leaf inventory contains exact source receipt fields; `.aris` contains the delegated request and full substantive review, not an invented complete host transcript. Do not publish either by default.
