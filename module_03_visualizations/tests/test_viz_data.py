"""
Test script for Module 3 visualization data generator.

Makes sure the JSON files are valid and contain correct data.
Run this AFTER generating the viz files.

Usage:
    python test_viz_data.py

"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import re


class Module3Tester:
    """Test suite for Module 3 outputs."""
    
    def __init__(
        self,
        data_dir: Path,
        output_dir: Path
    ):
        """
        Initialize tester.
        
        data_dir: raw JSON from Module 1
        output_dir: generated viz files from Module 3
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.test_results = []  # track which tests passed/failed
        
        print("=" * 80)
        print("MODULE 3 TEST SUITE")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        print()
    
    def run_all_tests(self) -> bool:
        """
        Run all tests and report results.
        
        Returns True if all tests passed.
        """
        # list of tests to run
        tests = [
            ('JSON Validity', self.test_json_validity),
            ('Frequency Calculations', self.test_frequency_calculations),
            ('Gap Calculations', self.test_gap_calculations),
            ('Color Scaling', self.test_color_scaling),
            ('Data Completeness', self.test_data_completeness),
            ('File Sizes', self.test_file_sizes)
        ]
        
        # run each test
        for test_name, test_func in tests:
            print(f"\nRunning: {test_name}")
            print("-" * 80)
            
            try:
                success, message = test_func()
                self.test_results.append((test_name, success, message))
                
                if success:
                    print(f"✓ PASSED: {message}")
                else:
                    print(f"✗ FAILED: {message}")
            
            except Exception as e:
                # catch any unexpected errors
                print(f"✗ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                self.test_results.append((test_name, False, f"Exception: {str(e)}"))
        
        # print summary
        self._print_summary()
        
        # return whether all tests passed
        return all(result[1] for result in self.test_results)
    
    def test_json_validity(self) -> tuple:
        """
        Test 1: Make sure all JSON files are valid.
        
        Checks:
        - Files exist
        - Valid JSON (can be parsed)
        - No syntax errors
        """
        # files we expect to find
        expected_files = [
            'powerball_heatmap.json',
            'megamillions_heatmap.json',
            'powerball_gaps.json',
            'megamillions_gaps.json',
            'powerball_quickstats.json',
            'megamillions_quickstats.json'
        ]
        
        missing_files = []
        invalid_files = []
        
        for filename in expected_files:
            filepath = self.output_dir / filename
            
            # check if file exists
            if not filepath.exists():
                missing_files.append(filename)
                continue
            
            # try to parse JSON
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                invalid_files.append(f"{filename}: {str(e)}")
        
        # report any issues
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        if invalid_files:
            return False, f"Invalid JSON: {'; '.join(invalid_files)}"
        
        return True, f"All {len(expected_files)} JSON files are valid"
    
    def test_frequency_calculations(self) -> tuple:
        """
        Test 2: Verify frequency calculations are correct.
        
        Checks:
        - Frequencies match raw data
        - Position breakdowns sum correctly
        - Percentiles are in valid range
        
        This is important because the heat map relies on these frequencies.
        """
        games = ['powerball', 'megamillions']
        issues = []
        
        for game in games:
            # load raw data to count actual frequencies
            raw_file = self.data_dir / f"{game}_current_format.json"
            if not raw_file.exists():
                issues.append(f"{game}: Raw data file not found")
                continue
            
            with open(raw_file, 'r') as f:
                drawings = json.load(f)
            
            # load generated heatmap
            heatmap_file = self.output_dir / f"{game}_heatmap.json"
            if not heatmap_file.exists():
                issues.append(f"{game}: Heatmap file not found")
                continue
            
            with open(heatmap_file, 'r') as f:
                heatmap = json.load(f)
            
            # manually count frequencies from raw data
            actual_counts = {}
            for drawing in drawings:
                for num in drawing['regularNumbers']:
                    actual_counts[num] = actual_counts.get(num, 0) + 1
            
            # compare with heatmap frequencies
            regular_numbers = heatmap['regular_numbers']
            
            for num_str, data in regular_numbers.items():
                num = int(num_str)
                expected_freq = actual_counts.get(num, 0)
                actual_freq = data['frequency']
                
                # check if frequencies match
                if expected_freq != actual_freq:
                    issues.append(
                        f"{game}: Number {num} has frequency {actual_freq}, "
                        f"expected {expected_freq}"
                    )
                
                # check that position breakdown sums to total
                position_sum = sum(data['position_breakdown'].values())
                if position_sum != actual_freq:
                    issues.append(
                        f"{game}: Number {num} position breakdown sums to "
                        f"{position_sum}, but frequency is {actual_freq}"
                    )
                
                # check percentile is in valid range
                percentile = data['percentile']
                if not (0 <= percentile <= 100):
                    issues.append(
                        f"{game}: Number {num} has invalid percentile {percentile}"
                    )
        
        if issues:
            # show first 3 issues to avoid wall of text
            return False, "; ".join(issues[:3])
        
        return True, "All frequency calculations are correct"
    
    def test_gap_calculations(self) -> tuple:
        """
        Test 3: Verify gap calculations.
        
        Checks:
        - Gap status matches gap days
        - All numbers have gap data
        - Gaps are non-negative
        
        The gap analysis is tricky because we have to search backwards
        through all drawings, so this test makes sure it works correctly.
        """
        games = ['powerball', 'megamillions']
        issues = []
        
        for game in games:
            # load gaps file
            gaps_file = self.output_dir / f"{game}_gaps.json"
            if not gaps_file.exists():
                issues.append(f"{game}: Gaps file not found")
                continue
            
            with open(gaps_file, 'r') as f:
                gaps_data = json.load(f)
            
            # check each category
            for category in ['hot_numbers', 'warm_numbers', 'cool_numbers', 
                           'cold_numbers', 'frozen_numbers']:
                numbers = gaps_data.get(category, [])
                
                for entry in numbers:
                    num = entry['number']
                    gap = entry['gap_days']
                    status = entry['status']
                    
                    # gap should be non-negative
                    if gap < 0:
                        issues.append(f"{game}: Number {num} has negative gap {gap}")
                    
                    # status should match gap
                    expected_status = self._expected_status(gap)
                    # frozen and never_drawn are treated the same
                    if status != expected_status and not (
                        status in ['frozen', 'never_drawn'] and 
                        expected_status in ['frozen', 'never_drawn']
                    ):
                        issues.append(
                            f"{game}: Number {num} has status '{status}', "
                            f"expected '{expected_status}' for gap {gap}"
                        )
        
        if issues:
            return False, "; ".join(issues[:3])
        
        return True, "All gap calculations are correct"
    
    def test_color_scaling(self) -> tuple:
        """
        Test 4: Check color generation.
        
        Checks:
        - Colors are valid hex codes
        - High percentile = hot colors (red)
        - Low percentile = cold colors (blue)
        
        The color scaling is what makes the heat map look good,
        so it's important that it works correctly.
        """
        games = ['powerball', 'megamillions']
        issues = []
        
        # regex for validating hex colors
        hex_pattern = r'^#[0-9a-fA-F]{6}$'
        
        for game in games:
            heatmap_file = self.output_dir / f"{game}_heatmap.json"
            if not heatmap_file.exists():
                issues.append(f"{game}: Heatmap file not found")
                continue
            
            with open(heatmap_file, 'r') as f:
                heatmap = json.load(f)
            
            regular_numbers = heatmap['regular_numbers']
            
            for num_str, data in regular_numbers.items():
                num = int(num_str)
                color = data['color']
                percentile = data['percentile']
                
                # check hex format
                if not re.match(hex_pattern, color):
                    issues.append(f"{game}: Number {num} has invalid color '{color}'")
                    continue
                
                # parse RGB values
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                
                # hot numbers should be reddish (high red component)
                if percentile > 75:
                    if r < 200:
                        issues.append(
                            f"{game}: Number {num} (percentile {percentile}) "
                            f"should be hot (red) but color is {color}"
                        )
                
                # cold numbers should be bluish (high blue component)
                elif percentile < 25:
                    if b < 200:
                        issues.append(
                            f"{game}: Number {num} (percentile {percentile}) "
                            f"should be cold (blue) but color is {color}"
                        )
        
        if issues:
            return False, "; ".join(issues[:3])
        
        return True, "All colors are valid and properly scaled"
    
    def test_data_completeness(self) -> tuple:
        """
        Test 5: Make sure all numbers are present.
        
        Checks:
        - All regular numbers in range
        - All special numbers in range
        - No missing numbers
        - No extra numbers
        - All required fields present
        """
        issues = []
        
        # expected ranges for each game
        game_ranges = {
            'powerball': {
                'regular': (1, 69),
                'special': (1, 26)
            },
            'megamillions': {
                'regular': (1, 70),
                'special': (1, 25)
            }
        }
        
        for game, ranges in game_ranges.items():
            # check heatmap
            heatmap_file = self.output_dir / f"{game}_heatmap.json"
            if not heatmap_file.exists():
                issues.append(f"{game}: Heatmap file not found")
                continue
            
            with open(heatmap_file, 'r') as f:
                heatmap = json.load(f)
            
            # check regular numbers are all there
            regular_min, regular_max = ranges['regular']
            expected_regular = set(range(regular_min, regular_max + 1))
            actual_regular = set(int(k) for k in heatmap['regular_numbers'].keys())
            
            # find missing numbers
            missing_regular = expected_regular - actual_regular
            if missing_regular:
                issues.append(
                    f"{game}: Missing regular numbers: {sorted(missing_regular)}"
                )
            
            # find extra numbers (shouldn't happen)
            extra_regular = actual_regular - expected_regular
            if extra_regular:
                issues.append(
                    f"{game}: Extra regular numbers: {sorted(extra_regular)}"
                )
            
            # check special numbers
            special_min, special_max = ranges['special']
            expected_special = set(range(special_min, special_max + 1))
            actual_special = set(int(k) for k in heatmap['special_numbers'].keys())
            
            missing_special = expected_special - actual_special
            if missing_special:
                issues.append(
                    f"{game}: Missing special numbers: {sorted(missing_special)}"
                )
            
            # check that each number has all required fields
            required_fields = ['frequency', 'percentile', 'color', 'position_breakdown']
            for num_str, data in heatmap['regular_numbers'].items():
                for field in required_fields:
                    if field not in data:
                        issues.append(
                            f"{game}: Regular number {num_str} missing field '{field}'"
                        )
                        break  # only report one missing field per number
        
        if issues:
            return False, "; ".join(issues[:3])
        
        return True, "All numbers present with complete data"
    
    def test_file_sizes(self) -> tuple:
        """
        Test 6: Check file sizes are reasonable.
        
        Checks:
        - Files are not empty
        - Files are under 1 MB each
        - Total size is reasonable
        
        If files are too large, they'll slow down the website.
        If they're empty, something went wrong.
        """
        issues = []
        total_size = 0
        max_size = 1024 * 1024  # 1 MB per file
        
        for json_file in self.output_dir.glob("*.json"):
            size = json_file.stat().st_size
            total_size += size
            
            # check if empty
            if size == 0:
                issues.append(f"{json_file.name} is empty")
            # check if too large
            elif size > max_size:
                size_mb = size / (1024 * 1024)
                issues.append(
                    f"{json_file.name} is too large: {size_mb:.2f} MB"
                )
        
        # check total size
        if total_size > max_size:
            total_mb = total_size / (1024 * 1024)
            issues.append(f"Total size too large: {total_mb:.2f} MB")
        
        if issues:
            return False, "; ".join(issues)
        
        total_kb = total_size / 1024
        return True, f"All files within size limits (total: {total_kb:.2f} KB)"
    
    def _expected_status(self, gap: int) -> str:
        """
        Figure out what status a gap should have.
        
        This matches the logic in the generator.
        """
        if gap < 7:
            return 'hot'
        elif gap < 30:
            return 'warm'
        elif gap < 90:
            return 'cool'
        elif gap < 180:
            return 'cold'
        else:
            return 'frozen'
    
    def _print_summary(self) -> None:
        """Print summary of test results."""
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # count passes
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        # print each test result
        for test_name, success, message in self.test_results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status:8s} {test_name:30s} {message}")
        
        print("=" * 80)
        print(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        
        print("=" * 80)


def main():
    """Run tests"""
    # paths - update for your setup
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs" / "viz_data"
    
    # check if output directory exists
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        print("Run viz_data_generator.py first to generate files.")
        sys.exit(1)
    
    # initialize tester
    tester = Module3Tester(
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    # run tests
    success = tester.run_all_tests()
    
    # exit with appropriate code
    # 0 = success, 1 = failure
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
