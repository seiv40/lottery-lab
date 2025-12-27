"""
Module 8: Version Checker
Check installed library versions and compatibility.

"""

import sys

print("\n--- MODULE 8: LIBRARY VERSION CHECKER ---")

print("\nPython Version:")
print(f"  {sys.version}")

print("\n--- REQUIRED LIBRARIES ---")

libraries = {
    'causal-learn': 'causallearn',
    'statsmodels': 'statsmodels',
    'dcor': 'dcor',
    'pyinform': 'pyinform',
    'networkx': 'networkx',
    'scipy': 'scipy',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scikit-learn': 'sklearn',
    'torch': 'torch'
}

missing = []
installed = []

for display_name, import_name in libraries.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        installed.append((display_name, version))
        print(f"[OK] {display_name:20s} {version}")
    except ImportError:
        missing.append(display_name)
        print(f"[MISSING] {display_name:20s} NOT INSTALLED")

print("\n--- CAUSAL-LEARN STRUCTURE CHECK ---")

if 'causal-learn' in [lib[0] for lib in installed]:
    try:
        from causallearn.search.ConstraintBased.PC import pc
        print("[OK] PC algorithm available")
    except ImportError:
        print("[MISSING] PC algorithm NOT available")
    
    try:
        from causallearn.search.ScoreBased.GES import ges
        print("[OK] GES algorithm available")
    except ImportError:
        print("[MISSING] GES algorithm NOT available")
    
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        print("[OK] FCI algorithm available")
    except ImportError:
        print("[MISSING] FCI algorithm NOT available")
    
    # check GFCI (multiple possible locations)
    gfci_available = False
    try:
        from causallearn.search.HybridCausal.GFCI import gfci
        print("[OK] GFCI algorithm available (HybridCausal)")
        gfci_available = True
    except ImportError:
        try:
            from causallearn.search.ConstraintBased.GFCI import gfci
            print("[OK] GFCI algorithm available (ConstraintBased)")
            gfci_available = True
        except ImportError:
            print("[WARNING] GFCI algorithm NOT available (will be skipped)")
    
    try:
        from causallearn.utils.cit import CIT
        print("[OK] Conditional independence tests available")
    except ImportError:
        print("[MISSING] Conditional independence tests NOT available")

print("\n--- SUMMARY ---")

if missing:
    print(f"\n[WARNING] {len(missing)} libraries missing:")
    for lib in missing:
        print(f"  - {lib}")
    print("\nInstall missing libraries with:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("\n[OK] All required libraries are installed")

if not gfci_available:
    print("\n[NOTE] GFCI algorithm will be skipped (not critical)")
    print("  The analysis will still run with PC, GES, and FCI algorithms")

print("\n--- VERSION CHECK COMPLETE ---")

if missing:
    print("\n[ACTION REQUIRED] Install missing libraries before running analysis")
    sys.exit(1)
else:
    print("\n[OK] System ready for Module 8 analysis")
    sys.exit(0)
