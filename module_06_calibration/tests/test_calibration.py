import json
import sys
from pathlib import Path


def test_calibration_files(outputs_dir):
    print("Testing calibration outputs...")
    
    required_files = [
        'calibration_report_powerball.json',
        'calibration_report_megamillions.json',
        'calibration_summary.json',
        'calibration_summary.csv'
    ]
    
    for filename in required_files:
        filepath = outputs_dir / filename
        assert filepath.exists(), f"Missing file: {filename}"
        print(f"  OK: {filename}")
    
    with open(outputs_dir / 'calibration_summary.json') as f:
        data = json.load(f)
        assert 'powerball' in data or 'megamillions' in data
        print("  OK: calibration_summary.json has valid structure")


def test_ensemble_files(outputs_dir):
    print("\nTesting ensemble outputs...")
    
    expected_files = [
        'ensemble_predictions_powerball.json',
        'ensemble_performance.json'
    ]
    
    for filename in expected_files:
        filepath = outputs_dir / filename
        if filepath.exists():
            print(f"  OK: {filename}")
            with open(filepath) as f:
                data = json.load(f)
                assert isinstance(data, dict)
                print(f"    Valid JSON structure")
        else:
            print(f"  Warning: {filename} not found")


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_calibration.py <module6_outputs_dir>")
        sys.exit(1)
    
    outputs_dir = Path(sys.argv[1])
    
    if not outputs_dir.exists():
        print(f"Error: Directory not found: {outputs_dir}")
        sys.exit(1)
    
    print("\n--- MODULE 6 OUTPUT VALIDATION ---")
    print(f"Testing directory: {outputs_dir}")
    print()
    
    try:
        test_calibration_files(outputs_dir)
        test_ensemble_files(outputs_dir)
        
        print("\n--- ALL TESTS PASSED ---")
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
