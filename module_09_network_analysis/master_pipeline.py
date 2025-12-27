"""
Module 9: Network Analysis - Master Pipeline

"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime


# Base directory
BASE_DIR = Path(r"C:\jackpotmath\lottery-lab")
MODULE_DIR = BASE_DIR / "modules" / "module_09_network_analysis"


class Module9Pipeline:
    """
    Master pipeline for Module 9 network analysis.
    """
    
    def __init__(self, lottery_name: str, n_null_samples: int = 100):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        lottery_name : str
            'powerball' or 'megamillions'
        n_null_samples : int
            Number of random graph samples for null models
        """
        self.lottery_name = lottery_name.lower()
        self.n_null_samples = n_null_samples
        
        if self.lottery_name not in ['powerball', 'megamillions']:
            raise ValueError(f"Unknown lottery: {lottery_name}")
        
        # define paths - outputs and figures go in module directory
        self.output_dir = MODULE_DIR / "outputs"
        self.figures_dir = MODULE_DIR / "figures"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = None
        self.step_times = {}
        
        print(f"MODULE 9 - NETWORK ANALYSIS PIPELINE")
        print(f"Lottery: {self.lottery_name.upper()}")
        print(f"Null samples: {n_null_samples}")
        print(f"Output directory: {self.output_dir}")
        print(f"Figures directory: {self.figures_dir}")
    
    def run_step(self, step_name: str, script_name: str, args: list = None):
        """
        Run a pipeline step (Python script).
        
        Parameters
        ----------
        step_name : str
            Name of the step for logging
        script_name : str
            Python script to execute
        args : list, optional
            Additional command-line arguments
        """
        print(f"STEP: {step_name}")
        print(f"Script: {script_name}")
        
        step_start = time.time()
        
        # Build command
        cmd = [sys.executable, script_name, self.lottery_name]
        
        if args:
            cmd.extend(args)
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Execute
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            
            step_elapsed = time.time() - step_start
            self.step_times[step_name] = step_elapsed
            
            print(f"\n[OK] {step_name} completed in {step_elapsed:.1f}s")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n[FAILED] {step_name} FAILED")
            print(f"Error: {e}")
            return False
        except FileNotFoundError:
            print(f"\n[FAILED] Script not found: {script_name}")
            return False
    
    def run_full_pipeline(self):
        """Execute complete Module 9 pipeline."""
        self.start_time = time.time()
        
        print(f"\nStarting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Graph Construction", "01_graph_construction.py", []),
            ("Network Metrics", "02_network_metrics.py", []),
            ("Null Models", "03_null_models.py", [str(self.n_null_samples)]),
            ("Statistical Testing", "04_statistical_tests.py", []),
            ("Advanced Analysis", "05_advanced_analysis.py", []),
            ("Visualizations", "06_visualizations.py", [])
        ]
        
        success_count = 0
        
        for step_name, script, args in steps:
            success = self.run_step(step_name, script, args)
            
            if success:
                success_count += 1
            else:
                print(f"\nPipeline STOPPED at step: {step_name}")
                break
        
        # Print summary
        self.print_summary(success_count, len(steps))
    
    def print_summary(self, completed: int, total: int):
        """Print pipeline execution summary."""
        total_elapsed = time.time() - self.start_time
        
        print("\n--- PIPELINE SUMMARY ---")
        
        print(f"\nLottery: {self.lottery_name.upper()}")
        print(f"Completed: {completed}/{total} steps")
        print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        
        if self.step_times:
            print("\nStep timings:")
            for step, elapsed in self.step_times.items():
                print(f"  {step:.<50} {elapsed:>6.1f}s")
        
        if completed == total:
            print("\n[OK] PIPELINE COMPLETED SUCCESSFULLY")
            print(f"\nOutputs saved to: {self.output_dir}")
            print(f"Figures saved to: {self.figures_dir}")
        else:
            print("\n[FAILED] PIPELINE INCOMPLETE")
        


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python master_pipeline.py <lottery_name> [n_null_samples]")
        print("  lottery_name: 'powerball' or 'megamillions'")
        print("  n_null_samples: number of null graph samples (default: 100)")
        sys.exit(1)
    
    lottery_name = sys.argv[1]
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    pipeline = Module9Pipeline(lottery_name, n_samples)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
