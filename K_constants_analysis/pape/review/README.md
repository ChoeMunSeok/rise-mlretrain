# 회의 기준 2015 Validation 인계

2012 fresh XGBoost, Source 2013-2014 (107,053 rows), Validation 2015 (57,452 rows), all 1,159 binary features.
72 runs = 12 monthly targets x 3 seeds x uniform/dre. Test 2016-2018 was NOT evaluated.
validation_review_summary.json and monthly_comparison.csv are evaluation/review files, not operational GAIN/RCCDA input.
Operational inputs are gain_inputs.parquet and rccda_inputs.json. Select one seed and one method, use readiness/warnings.
Original research execution and Git port have separate code revisions. January uniform/DRE port replay matched exactly.
The original model was freshly trained; identical weights from a fixed seed do not mean checkpoint reuse.
Annual Source additions are a guarded handoff plan. Test execution belongs to later integration.
The full bundle contains original model/export/result manifests. Historical absolute paths in those manifests are provenance;
use bundle_manifest logical_paths to find packaged files after moving the bundle.
Do not commit any of these results, data sidecars, model or ZIP archives to Git.
