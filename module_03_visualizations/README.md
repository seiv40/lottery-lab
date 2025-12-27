# Module 3: Visualization Data Generator

Generates JSON files for the lottery visualizations on JackpotMath.com. Creates heat maps, gap analysis, and quick stats that the website uses.

## How to Run

```bash
# Generate all 6 JSON files
python modules/module_03_visualizations/viz_data_generator.py

# Run tests
python modules/module_03_visualizations/tests/test_viz_data.py
```

## Output Files

Creates 6 JSON files (total size about 150-200 KB):
- `powerball_heatmap.json` - Frequency data with colors
- `megamillions_heatmap.json`
- `powerball_gaps.json` - Hot/cold number analysis
- `megamillions_gaps.json`
- `powerball_quickstats.json` - Dashboard summary
- `megamillions_quickstats.json`

## What's In Each File

**Heat Maps:** Number frequencies with color codes (red = hot, blue = cold)

**Gap Analysis:** Numbers categorized by how recently drawn (hot/warm/cool/cold/frozen)

**Quick Stats:** Top 10 lists, averages, and dashboard data
