# Monitoring data

The three workbooks in `raw/` are the public inputs from repository revision
[`86e3ac4`](https://github.com/gao1718/S2SMAG/tree/86e3ac477934db0500ed851607e98660ff1070f8).
Their contents are unchanged.

| Workbook | Variable | Unit | SHA-256 |
| --- | --- | --- | --- |
| `settlement.xlsx` | Cumulative settlement | mm | `7b113213d5c7f1dff6db92261d7514aaec313cbc6e6eeecd0be8cfca9a6e2d438` |
| `settlement_rate.xlsx` | Settlement increment over one day | mm/day | `e1199cc9ebb9ffa506a6f9456c3aa6394ca95ed12dcc8fd22fecf047bcebeb7c` |
| `vacuum.xlsx` | Vacuum pressure magnitude | kPa | `5d328d7ec9d618e1a3c226ea5409f0787c365cd66a7d300a1711637d1ab5504d` |

The first column in each workbook is the date. The records cover 143 daily observations
from 24 December 2019 through 14 May 2020. Settlement uses a negative-down sign convention.
The preparation script uses the following correspondence for the vacuum columns.

| Settlement point | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vacuum column suffix | MM1 | MM2 | MM3 | MM6 | MM8 | MM9 | MM10 | MM11 | MM12 |

Run `python -m original_data.prepare_revision --output-dir original_data/processed/c1_c9`
from the repository root. Preparation validates the daily dates and column mapping,
fits the min–max scale on C2–C8, and makes 10-day input/10-day output windows at a
one-day stride within each point. It writes the three partitions, the scale parameters,
source and output checksums, and the C1 observation reference used for evaluation.
The nine monitoring points share the same calendar. C1 and C9 observations only undergo
the transform learned from C2–C8.
