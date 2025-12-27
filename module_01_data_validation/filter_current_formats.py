#!/usr/bin/env python3
"""
Module 1: Data Validation and Format Filtering

Filters lottery data to keep only current format drawings.
Powerball: Oct 7, 2015+ (5/69 + 1/26)
Mega Millions: Oct 31, 2017+ (5/70 + 1/25 or 1/24)

"""

import json
from datetime import datetime
from collections import defaultdict

# Cutoff dates for when current formats started
# I looked these up on the official lottery websites
POWERBALL_CUTOFF = "2015-10-07"
MEGAMILLIONS_CUTOFF = "2017-10-31"
MEGABALL_CHANGE = "2025-04-08"  # when they changed mega ball to 1-24

# tried using datetime objects first but string comparison works fine for ISO dates
# dt_cutoff = datetime.fromisoformat(POWERBALL_CUTOFF)
# decided against it - unnecessary complexity


def filter_powerball_data(drawings):
    """Filter Powerball to current format (5/69 + 1/26)"""
    filtered = []
    removed_count = 0
    invalid_numbers = 0
    
    for draw in drawings:
        date = draw['dateISO']
        
        # skip old format drawings (before Oct 2015)
        if date < POWERBALL_CUTOFF:
            removed_count += 1
            continue
            
        # get the numbers from this drawing
        regular_nums = draw.get('regularNumbers', [])
        powerball = draw.get('specialNumber')
        
        # now validate that the numbers are in the correct range
        is_valid = True
        
        # should have exactly 5 regular numbers
        if len(regular_nums) != 5:
            is_valid = False
            invalid_numbers += 1
        
        # each regular number should be 1-69
        for num in regular_nums:
            if not (1 <= num <= 69):
                is_valid = False
                invalid_numbers += 1
                break  # no point checking the rest
        
        # powerball should be 1-26
        if powerball is None or not (1 <= powerball <= 26):
            is_valid = False
            invalid_numbers += 1
        
        # only keep valid drawings
        if is_valid:
            filtered.append(draw)
        else:
            removed_count += 1
    
    # calculate date range for the report
    if filtered:
        dates = [d['dateISO'] for d in filtered]
        date_range = {'start': min(dates), 'end': max(dates)}
    else:
        date_range = {}  # shouldn't happen but just in case
    
    # return filtered data + stats for reporting
    stats = {
        'total_input': len(drawings),
        'kept': len(filtered),
        'removed': removed_count,
        'invalid_numbers': invalid_numbers,
        'date_range': date_range
    }
    
    return filtered, stats


def filter_megamillions_data(drawings):
    """Filter Mega Millions to current format (5/70 + 1/25 or 1/24)"""
    filtered = []
    removed_count = 0
    invalid_numbers = 0
    
    # track how many drawings fall into each mega ball range
    # this will be useful to know
    megaball_1_25 = 0
    megaball_1_24 = 0
    
    for draw in drawings:
        date = draw['dateISO']
        
        # skip old format drawings (before Oct 2017)
        if date < MEGAMILLIONS_CUTOFF:
            removed_count += 1
            continue
        
        regular_nums = draw.get('regularNumbers', [])
        megaball = draw.get('specialNumber')
        
        # mega ball range changed in April 2025
        # need to check which range this drawing should use
        if date >= MEGABALL_CHANGE:
            # after April 8, 2025: mega ball is 1-24
            megaball_max = 24
            megaball_range = '1-24'
        else:
            # before April 8, 2025: mega ball is 1-25
            megaball_max = 25
            megaball_range = '1-25'
        
        # validate the numbers
        is_valid = True
        
        # should have exactly 5 regular numbers
        if len(regular_nums) != 5:
            is_valid = False
            invalid_numbers += 1
        
        # each regular number should be 1-70
        for num in regular_nums:
            if not (1 <= num <= 70):
                is_valid = False
                invalid_numbers += 1
                break  # no point checking rest if one is invalid
        
        # mega ball should be within the correct range for this date
        if megaball is None or not (1 <= megaball <= megaball_max):
            is_valid = False
            invalid_numbers += 1
        
        if is_valid:
            # add metadata about which range this drawing uses
            # might be useful for later analysis if the change affects patterns
            draw['_megaball_range'] = megaball_range
            filtered.append(draw)
            
            # track the counts
            if megaball_range == '1-24':
                megaball_1_24 += 1
            else:
                megaball_1_25 += 1
        else:
            removed_count += 1
    
    # calculate date range for the report
    if filtered:
        dates = [d['dateISO'] for d in filtered]
        date_range = {'start': min(dates), 'end': max(dates)}
    else:
        date_range = {}  # shouldn't happen
    
    # return filtered data + detailed stats
    stats = {
        'total_input': len(drawings),
        'kept': len(filtered),
        'removed': removed_count,
        'invalid_numbers': invalid_numbers,
        'date_range': date_range,
        'megaball_1_25_count': megaball_1_25,
        'megaball_1_24_count': megaball_1_24
    }
    
    return filtered, stats


def main():
    import os
    
    # make sure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n--- Module 1: Filtering to Current Formats ---")
    
    print("\nLoading raw data...")
    
    # tried using pandas first but json is simpler for this
    # import pandas as pd
    # pb_df = pd.read_json('data/powerball.json')
    # json module works fine and doesn't need extra dependency
    
    with open('data/powerball.json', 'r') as f:
        powerball_raw = json.load(f)
    print(f"Loaded {len(powerball_raw)} Powerball drawings")
    
    with open('data/megamillions.json', 'r') as f:
        megamillions_raw = json.load(f)
    print(f"Loaded {len(megamillions_raw)} Mega Millions drawings")
    
    # Filter Powerball to current format
    print("\n--- Filtering Powerball (Oct 7, 2015+) ---")
    print("\n--- Format: 5/69 + 1/26 ---")

    
    pb_filtered, pb_stats = filter_powerball_data(powerball_raw)
    
    # print results
    print(f"Kept: {pb_stats['kept']} drawings")
    print(f"Removed: {pb_stats['removed']} drawings")
    if pb_stats['date_range']:
        print(f"Date range: {pb_stats['date_range']['start']} to {pb_stats['date_range']['end']}")
    
    # Filter Mega Millions to current format
    print("\n--- Filtering Mega Millions (Oct 31, 2017+) ---")
    print("\n--- Format: 5/70 + 1/25 (or 1/24 after Apr 8, 2025) ---")
    
    mm_filtered, mm_stats = filter_megamillions_data(megamillions_raw)
    
    print(f"Kept: {mm_stats['kept']} drawings")
    print(f"Removed: {mm_stats['removed']} drawings")
    if mm_stats['date_range']:
        print(f"Date range: {mm_stats['date_range']['start']} to {mm_stats['date_range']['end']}")
    
    # show the mega ball range split - good to know
    print(f"\nMega Ball range split:")
    print(f"  1-25: {mm_stats['megaball_1_25_count']} drawings")
    print(f"  1-24: {mm_stats['megaball_1_24_count']} drawings")
    
    # Save filtered data to processed folder
    print("\n--- Saving... ---")
    
    with open('data/processed/powerball_current_format.json', 'w') as f:
        json.dump(pb_filtered, f, indent=2)
    print("Saved: powerball_current_format.json")
    
    with open('data/processed/megamillions_current_format.json', 'w') as f:
        json.dump(mm_filtered, f, indent=2)
    print("Saved: megamillions_current_format.json")
    
    # Save a detailed report for documentation
    # this will be useful when I write up the methodology
    report = {
        'powerball': {
            'total_input': pb_stats['total_input'],
            'kept': pb_stats['kept'],
            'removed': pb_stats['removed'],
            'format': '5/69 + 1/26',
            'cutoff_date': POWERBALL_CUTOFF,
            'date_range': pb_stats['date_range']
        },
        'megamillions': {
            'total_input': mm_stats['total_input'],
            'kept': mm_stats['kept'],
            'removed': mm_stats['removed'],
            'format': '5/70 + 1/25 or 1/24',
            'cutoff_date': MEGAMILLIONS_CUTOFF,
            'megaball_change_date': MEGABALL_CHANGE,
            'date_range': mm_stats['date_range'],
            'megaball_distribution': {
                '1-25': mm_stats['megaball_1_25_count'],
                '1-24': mm_stats['megaball_1_24_count']
            }
        },
        'summary': {
            'total_kept': pb_stats['kept'] + mm_stats['kept'],
            'total_removed': pb_stats['removed'] + mm_stats['removed']
        }
    }
    
    with open('data/processed/filtering_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved: filtering_report.json")
    
    # Print final summary
    print("\n--- Done! ---")
    print(f"Powerball: {pb_stats['kept']} drawings")
    print(f"Mega Millions: {mm_stats['kept']} drawings")
    print(f"Total: {pb_stats['kept'] + mm_stats['kept']} drawings")
    print(f"\nRemoved {pb_stats['removed'] + mm_stats['removed']} old format drawings")
    print("\nReady for Module 2")


if __name__ == '__main__':
    main()
    
    # TODO: add validation check to make sure we have enough drawings
    # need at least 100 per lottery for the chi-square tests in module 3
    # otherwise statistical power will be too low
