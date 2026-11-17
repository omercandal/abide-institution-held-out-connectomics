# ABIDE I institution-held-out connectomics

Release candidate prepared for author review on 17 November 2026.
Author: Ömer Faruk Candal, Independent Researcher, Samsun, Türkiye.
ORCID: https://orcid.org/0000-0001-9451-4173

## Scientific scope

This package accompanies **Incremental Value of Resting-State Functional
Connectivity Beyond Age, Sex, and Motion for Autism Classification: An
Institution-Held-Out ABIDE I Study**. The primary cohort contains 830
participants (376 ASD, 454 controls), with 20 acquisition-subcohort labels
grouped into 17 institutions. The primary endpoint is equal-institution
mean ROC-AUC. Logistic-regression AUCs are 0.528 for age/sex/mean motion,
0.665 for connectivity, and 0.654 for combined features. The paired
combined-minus-reference increment is 0.126 (conditional 95% interval
0.070 to 0.173). These are research estimates, not a clinical diagnostic tool.

## Included and omitted materials

Included: four original scientific source scripts, the archived analysis
plan, pinned package versions, aggregate results, 4 main and 4 supplementary
figures in PNG/PDF/TIFF formats, source provenance, and release checksums.

Not redistributed: original MRI or ROI time series, phenotype records,
participant feature matrices, participant predictions, split membership,
fold checkpoints, the original executed notebook with embedded outputs,
private working logs, or the unpublished manuscript. Those data and working
records must not be added to the public repository by a broad directory upload.
This release is sufficient to inspect aggregate conclusions and contains
source code to reconstruct participant-level working files from the original
public provider, subject to its terms. It is not a full participant-level
audit archive. Individual-prediction checks and exact uncertainty recovery
require a permitted source-data rerun or access arranged by the author.

## Quick aggregate verification

No third-party dependency is needed for this check:

```bash
python validate_release.py
```

This verifies file integrity, compares aggregate AUCs to the stored
institution scores, checks cohort totals, and checks the three paired
point estimates. It does **not** refit models or validate clinical utility.

## Reconstruct the analysis from the provider

The original environment record used Python 3.12.14. The original platform
was a cloud Python runtime; the notebook was executed using in-process
IPython. It was not an executed Google Colab account session. Python and
package versions are recorded in provenance/software_environment.json.

Use a new checkout and a dedicated environment. Install requirements.txt,
then run:

```bash
python run_analysis.py --fetch-public-data
```

This explicitly requests retrieval of public PCP ABIDE I C-PAC filt_noglobal
Harvard–Oxford ROI derivatives and the phenotype CSV. It creates local,
git-ignored data/, results/, and checkpoints/ directories. The run may take
substantial time and requires network access to the data provider. Confirm
your intended use and source-data terms first. Existing fold checkpoints
cause the runner to stop unless --resume is supplied; --resume is checkpoint
recovery, not an independent fresh refit.

The runner executes cohort construction, all seven variant definitions,
aggregation, statistics, and figures. It intentionally does not regenerate
the obsolete original manuscript or create the original execution notebook.
The four scientific source scripts are byte-identical to the verified
archived originals. The new runner only initializes directories, invokes
them, and aggregates their output in the archived variant order.

## Interpretation limits

Bootstrap intervals condition on already fitted outer predictions and omit
model-refitting uncertainty. Sign-flip diagnostics are exploratory. The pilot
informed settings and was not an independent validation cohort. ABIDE II
validation, clinical calibration, and clinical utility testing are absent.
Complete-case sensitivity uses 639 participants with 15 informative
institutions; strict manual QC uses 568 with 13 informative institutions.
Source cohort_flow.json reports 20 raw SITE_ID labels, not 20 validation
institutions; institution_validation_groups is 17.

The figure files are archived outputs. Figure_S4 uses connected sensitivity
variants, which do not form an ordered trajectory. Uncertainty and denominators
must be read from aggregate_results/summary_with_CI.csv and the manuscript.

## Sources and rights

ABIDE I: https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html
PCP: https://preprocessed-connectomes-project.org/abide/
ABIDE I publication: https://doi.org/10.1038/mp.2013.78

The ABIDE source page identifies CC BY-NC-SA 3.0 and non-commercial research
conditions. These source rights are not replaced by this repository.
No source participant records are redistributed here. Original analysis code
and associated usage documentation are licensed under MIT; see LICENSE and
LICENSING.md for the exact scope. The license does not relicense ABIDE/PCP
data, aggregate results, figures, or third-party materials.

Public repository:
https://github.com/omercandal/abide-institution-held-out-connectomics
The initial reviewed scientific package is commit
1d655a97bb0f0c6b636aff44cfd2a845cd327208. No DOI has been minted.

## Attribution and AI assistance

Please retain attribution to ABIDE participants, contributing investigators,
and the Preprocessed Connectomes Project. Original ABIDE resource support
is acknowledged by the data providers. This does not imply financial support
to the present author.

ChatGPT Work (OpenAI) assisted with code development, cloud execution,
checking, interpretation support, literature retrieval, figure preparation,
and writing. Exact historical model identifiers and product build numbers
were not retained in the supplied analysis archive. The archived Python
environment is not an AI model-version record. Authors remain responsible
for scientific review. See docs/AI_USE_RECORD.json.
