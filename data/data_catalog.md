raw/ → contains your unaltered source datasets (e.g., equities.csv, commodities.csv, macro_indicators.xlsx).

processed/ → outputs from the pipeline (aligned, merged, feature-enriched data ready for modeling).

data_catalog.md → a single Markdown file documenting:
	•	Data source links (Yahoo Finance, FRED, Quandl, etc.)
	•	Frequency (daily/monthly)
	•	Variables (symbols, fields, units)
	•	Transformation summary (e.g., “log returns,” “z-scored CPI”)