# Module 1: Data Validation and Format Filtering

Filters raw lottery data to keep only current-format drawings. This matters because mixing old and new formats breaks statistical assumptions (different number ranges = biased frequency tests).

## What It Does

**Powerball:** Keeps drawings from Oct 7, 2015 onwards (5/69 + 1/26 format)

**Mega Millions:** Keeps drawings from Oct 31, 2017 onwards (5/70 + 1/25 format)

Removes about 600 Powerball and 700 Mega Millions drawings, leaving roughly 1,200-1,500 per lottery.

## How to Run

```bash
python modules/module_01_data_validation/filter_current_formats.py
```

## Input/Output

**Input:**
- `data/powerball.json`
- `data/megamillions.json`

**Output:**
- `data/processed/powerball_current_format.json`
- `data/processed/megamillions_current_format.json`
- `data/processed/filtering_report.json`

## Why Current Formats Only?

Initially I kept all historical data but ran into problems:
- Different number ranges bias frequency analysis
- Chi-square tests become invalid
- ML models see format changes as patterns

Smaller sample size but cleaner analysis.

**Note:** Mega Millions changed from 1-25 to 1-24 Mega Ball in April 2025. I kept both ranges since they're close enough. Added a `_megaball_range` field to track which is which in case it causes problems later.
