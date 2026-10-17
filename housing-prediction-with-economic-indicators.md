# Housing Prediction with Economic Indicators

## Data
King County Sales; FRED MORTGAGE30US, CUUSA423SA0, SEAT653URN and PCPI53033

## Tools
Python, rdata, pandas, XGBoost, PyTorch-based TabNet and Jupyter

## Work completed
- Built a 229,046-row, 55-column housing dataset by joining mortgage rates, Seattle CPI, Seattle unemployment and King County per-capita income to sales records.
- Converted R-format records, aligned different reporting frequencies to sale dates and checked indicator coverage. Extended the model notebook with structural/full-feature comparisons, fold metrics, feature importance, residual plots and out-of-fold predictions.

## Evidence and limitations
The local TabNet run used a shorter configuration and is not claimed to reproduce the dissertation scores.

Raw datasets and private operational records are not distributed.

## Source files
- [join_external_economic_data.py](housing-prediction-with-economic-indicators__join_external_economic_data.py)
- [KingCounty_Housing_Prediction_with_Economic_Indicators.ipynb](housing-prediction-with-economic-indicators__KingCounty_Housing_Prediction_with_Economic_Indicators.ipynb)
- [build_kingcounty_finance_join.py](housing-prediction-with-economic-indicators__build_kingcounty_finance_join.py)
