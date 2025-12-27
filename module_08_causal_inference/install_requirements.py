"""
Module 8: Installation Helper

Run with: python install_requirements.py

"""

import subprocess
import sys

print("\n--- MODULE 8: AUTOMATED INSTALLATION ---")

# required packages
packages = [
    'causal-learn>=0.1.3.3',
    'statsmodels>=0.14.0',
    'dcor>=0.6',
    'pyinform>=0.1.0',
    'networkx>=3.0',
    'scipy>=1.10.0',
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'scikit-learn>=1.3.0',
    'torch>=2.0.0'
]

print("\nInstalling required packages...")
print("-" * 70)

failed = []
succeeded = []

for package in packages:
    package_name = package.split('>=')[0]
    print(f"\nInstalling {package_name}...", end=" ", flush=True)
    
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package, '--quiet'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("[OK]")
        succeeded.append(package_name)
    except subprocess.CalledProcessError:
        print("[FAILED]")
        failed.append(package_name)

print("\n--- INSTALLATION SUMMARY ---")

print(f"\n[OK] Successfully installed: {len(succeeded)}/{len(packages)}")
for pkg in succeeded:
    print(f"  - {pkg}")

if failed:
    print(f"\n[FAILED] Failed to install: {len(failed)}")
    for pkg in failed:
        print(f"  - {pkg}")
    print("\nTry installing failed packages manually:")
    for pkg in failed:
        print(f"  pip install {pkg}")
else:
    print("\n[OK] All packages installed successfully!")

print("\n--- NEXT STEPS ---")

print("\n1. Run version checker:")
print("   python check_versions.py")

print("\n2. Update config.py with your data paths")

print("\n3. Run the analysis:")
print("   python module8_complete_analysis.py")

if failed:
    print("\n[WARNING] Some packages failed to install")
    print("  Please install them manually before proceeding")
    sys.exit(1)
else:
    print("\n[OK] Ready to run Module 8!")
    sys.exit(0)
