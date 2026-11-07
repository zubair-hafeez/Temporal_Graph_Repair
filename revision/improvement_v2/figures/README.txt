recovery_damage.pdf and .png are produced by present.py from analysis.json.

Each marker is the mean over three allocation/fault seeds. Color encodes the
verified-history budget; circles use eight real-history corruption variants,
and triangles additionally use generated workflows. The two points per line
use the fixed .99 and .95 calibration precision targets. Lines are visual guides,
not interpolated or guaranteed achievable repair policies. No operating point
was selected using these test outcomes for deployment.

Horizontal: newly incorrect events per 1,000 originally correct events.
Vertical: percentage of originally incorrect events fully recovered.
Both metrics expand identity clusters to account for false-merge contamination.
The test roots are reused source cohorts; three seeds are not three institutions.
All underlying counts, including clean-input and pipeline stress outcomes, are
available in per_trial.csv one directory above.
