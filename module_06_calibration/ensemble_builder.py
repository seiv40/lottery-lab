"""
Module 6: Ensemble Builder

Combines multiple models' predictions using different weighting strategies.

Usage:
    python ensemble_builder.py <powerball_dir> <megamillions_dir> <outputs_dir>
    
Example:
    python ensemble_builder.py ../../output/powerball ../../output/megamillions module6_outputs

"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys


class EnsembleBuilder:
    """Build ensemble predictions from multiple models."""
    
    def __init__(self, powerball_dir: str, megamillions_dir: str, outputs_dir: str):
        """Initialize ensemble builder."""
        self.powerball_dir = Path(powerball_dir)
        self.megamillions_dir = Path(megamillions_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # models that provide prediction intervals
        self.models_with_intervals = [
            'bnn',
            'deepset',
            'deepset_v2',
            'transformer_bayeshead_baseline',
            'transformer_bayeshead_heavy',
            'transformer_bayeshead_hetero',
            'transformer_bayeshead_hetero_heavy'
        ]
        
        print("=" * 80)
        print("MODULE 6: ENSEMBLE BUILDER")
        print("=" * 80)
        print(f"Powerball dir: {self.powerball_dir}")
        print(f"Mega Millions dir: {self.megamillions_dir}")
        print(f"Output dir: {self.outputs_dir}")
        print()
    
    def load_model_data(self, model_name: str, lottery: str) -> Optional[Dict]:
        """Load predictions for a model."""
        if lottery == 'powerball':
            data_dir = self.powerball_dir
            suffix = 'pb'
        else:
            data_dir = self.megamillions_dir
            suffix = 'mm'
        
        pred_file = data_dir / f"{model_name}_predictive_{suffix}.json"
        perf_file = data_dir / f"{model_name}_perf_{suffix}.json"
        
        if not pred_file.exists() or not perf_file.exists():
            return None
        
        try:
            with open(pred_file) as f:
                pred = json.load(f)
            with open(perf_file) as f:
                perf = json.load(f)
            
            return {
                'predictions': pred,
                'performance': perf,
                'model_name': model_name
            }
        except Exception as e:
            print(f"  Warning: Error loading {model_name}: {e}")
            return None
    
    def build_mean_ensemble(self, models: List[Dict]) -> Dict:
        """Simple mean ensemble."""
        y_preds = []
        for m in models:
            pred = m['predictions']
            # check for both y_pred and y_mean
            if 'y_pred' in pred:
                y_preds.append(np.array(pred['y_pred']))
            elif 'y_mean' in pred:
                y_preds.append(np.array(pred['y_mean']))
        
        if len(y_preds) == 0:
            return {}
        
        # align to minimum length (handle different test set sizes)
        min_len = min(len(arr) for arr in y_preds)
        y_preds_aligned = [arr[:min_len] for arr in y_preds]
        
        return {'y_pred': np.mean(y_preds_aligned, axis=0).tolist()}
    
    def build_variance_weighted_ensemble(self, models: List[Dict]) -> Dict:
        """Weight by inverse variance (confident models get higher weight)."""
        y_preds = []
        weights = []
        
        for m in models:
            pred = m['predictions']
            # need both predictions and std
            y_pred_val = pred.get('y_pred') or pred.get('y_mean')
            y_std_val = pred.get('y_std')
            
            if y_pred_val is None or y_std_val is None:
                continue
            
            y_preds.append(np.array(y_pred_val))
            std = np.array(y_std_val)
            
            # weight = 1 / variance
            # add small epsilon to avoid division by zero
            var = std ** 2 + 1e-8
            weights.append(1.0 / var)
        
        if len(y_preds) == 0:
            return {}
        
        # align to minimum length
        min_len = min(len(arr) for arr in y_preds)
        y_preds = [arr[:min_len] for arr in y_preds]
        weights = [arr[:min_len] for arr in weights]
        
        y_preds = np.array(y_preds)  # shape: (n_models, n_samples)
        weights = np.array(weights)  # shape: (n_models, n_samples)
        
        # normalize weights so they sum to 1 for each sample
        weights = weights / weights.sum(axis=0, keepdims=True)
        
        # weighted average
        y_pred = (y_preds * weights).sum(axis=0)
        
        return {'y_pred': y_pred.tolist()}
    
    def build_calibration_weighted_ensemble(self, models: List[Dict], 
                                           calibration_report: Dict) -> Dict:
        """Weight by calibration quality (well-calibrated models get higher weight)."""
        y_preds = []
        weights = []
        
        for m in models:
            model_name = m['model_name']
            pred = m['predictions']
            
            # check for both y_pred and y_mean
            y_pred_val = pred.get('y_pred') or pred.get('y_mean')
            if y_pred_val is None:
                continue
            
            # get calibration metrics
            if model_name not in calibration_report:
                continue
            
            cal = calibration_report[model_name]
            
            # handle ece being either a dict or a float
            ece_val = cal.get('ece', 1.0)
            if isinstance(ece_val, dict):
                # if it's a dict, try to get a mean/value field
                ece = ece_val.get('mean', ece_val.get('value', 1.0))
            else:
                ece = ece_val
            
            # weight = 1 / (ECE + epsilon)
            # lower ECE = better calibration = higher weight
            weight = 1.0 / (float(ece) + 0.01)
            
            y_preds.append(np.array(y_pred_val))
            weights.append(weight)
        
        if len(y_preds) == 0:
            return {}
        
        # align to minimum length
        min_len = min(len(arr) for arr in y_preds)
        y_preds = [arr[:min_len] for arr in y_preds]
        
        y_preds = np.array(y_preds)
        weights = np.array(weights)
        
        # normalize weights
        weights = weights / weights.sum()
        
        # weighted average
        y_pred = (y_preds.T * weights).T.sum(axis=0)
        
        return {'y_pred': y_pred.tolist()}
    
    def build_quantile_ensemble(self, models: List[Dict]) -> Dict:
        """Conservative ensemble using median and min/max bounds."""
        y_preds = []
        
        for m in models:
            pred = m['predictions']
            # check for both y_pred and y_mean
            y_pred_val = pred.get('y_pred') or pred.get('y_mean')
            if y_pred_val is not None:
                y_preds.append(np.array(y_pred_val))
        
        if len(y_preds) == 0:
            return {}
        
        # align to minimum length
        min_len = min(len(arr) for arr in y_preds)
        y_preds = [arr[:min_len] for arr in y_preds]
        
        y_preds = np.array(y_preds)
        
        return {
            'y_pred': np.median(y_preds, axis=0).tolist(),
            'y_lower': np.min(y_preds, axis=0).tolist(),
            'y_upper': np.max(y_preds, axis=0).tolist()
        }
    
    def evaluate_ensemble(self, ensemble: Dict, y_true: np.ndarray) -> Dict:
        """Calculate RMSE for ensemble."""
        if 'y_pred' not in ensemble or len(ensemble['y_pred']) == 0:
            return {'rmse': None}
        
        y_pred = np.array(ensemble['y_pred'])
        
        # align lengths (use minimum)
        n = min(len(y_pred), len(y_true))
        y_pred = y_pred[:n]
        y_true = y_true[:n]
        
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        return {'rmse': float(rmse), 'n_samples': n}
    
    def build_lottery_ensembles(self, lottery: str) -> Dict:
        """Build all ensembles for a lottery."""
        print(f"\nBuilding ensembles for {lottery.upper()}...")
        
        # load all models
        models = []
        for model_name in self.models_with_intervals:
            data = self.load_model_data(model_name, lottery)
            if data:
                models.append(data)
                print(f"  Loaded: {model_name}")
        
        if len(models) == 0:
            print(f"  No models with predictions found for {lottery}")
            return {}
        
        # get y_true from first model (check both predictions and performance)
        y_true = None
        for m in models:
            if 'y_true' in m['predictions']:
                y_true = np.array(m['predictions']['y_true'])
                break
            elif 'y_true' in m['performance']:
                y_true = np.array(m['performance']['y_true'])
                break
        
        if y_true is None:
            print(f"  Warning: No y_true found for {lottery}")
            return {}
        
        # load calibration report if exists
        cal_file = self.outputs_dir / f"calibration_report_{lottery}.json"
        calibration_report = {}
        if cal_file.exists():
            with open(cal_file) as f:
                calibration_report = json.load(f)
        
        # build ensembles
        results = {}
        
        # mean ensemble
        ens = self.build_mean_ensemble(models)
        if ens:
            perf = self.evaluate_ensemble(ens, y_true)
            results['mean'] = {'predictions': ens, 'performance': perf}
            print(f"  Mean ensemble: RMSE={perf.get('rmse', 'N/A'):.4f}")
        
        # variance-weighted
        ens = self.build_variance_weighted_ensemble(models)
        if ens:
            perf = self.evaluate_ensemble(ens, y_true)
            results['variance_weighted'] = {'predictions': ens, 'performance': perf}
            print(f"  Variance-weighted: RMSE={perf.get('rmse', 'N/A'):.4f}")
        
        # calibration-weighted
        if calibration_report:
            ens = self.build_calibration_weighted_ensemble(models, calibration_report)
            if ens:
                perf = self.evaluate_ensemble(ens, y_true)
                results['calibration_weighted'] = {'predictions': ens, 'performance': perf}
                print(f"  Calibration-weighted: RMSE={perf.get('rmse', 'N/A'):.4f}")
        
        # quantile
        ens = self.build_quantile_ensemble(models)
        if ens:
            perf = self.evaluate_ensemble(ens, y_true)
            results['quantile'] = {'predictions': ens, 'performance': perf}
            print(f"  Quantile ensemble: RMSE={perf.get('rmse', 'N/A'):.4f}")
        
        return results
    
    def run(self):
        """Build all ensembles."""
        print("\n--- BUILDING ENSEMBLES ---")
        
        # powerball
        pb_results = self.build_lottery_ensembles('powerball')
        
        # save powerball
        if pb_results:
            out_file = self.outputs_dir / "ensemble_predictions_powerball.json"
            with open(out_file, 'w') as f:
                json.dump(pb_results, f, indent=2)
            print(f"\nSaved: {out_file}")
        
        # mega millions
        mm_results = self.build_lottery_ensembles('megamillions')
        
        # save mega millions
        if mm_results:
            out_file = self.outputs_dir / "ensemble_predictions_megamillions.json"
            with open(out_file, 'w') as f:
                json.dump(mm_results, f, indent=2)
            print(f"Saved: {out_file}")
        
        # save combined performance
        performance = {
            'powerball': {k: v['performance'] for k, v in pb_results.items()} if pb_results else {},
            'megamillions': {k: v['performance'] for k, v in mm_results.items()} if mm_results else {}
        }
        
        perf_file = self.outputs_dir / "ensemble_performance.json"
        with open(perf_file, 'w') as f:
            json.dump(performance, f, indent=2)
        print(f"Saved: {perf_file}")
        
        # print summary
        self.print_summary(pb_results, mm_results)
        
        print("\n--- ENSEMBLE BUILDING COMPLETE ---")
    
    def print_summary(self, pb_results: Dict, mm_results: Dict):
        """Print performance summary."""
        print("\n--- ENSEMBLE PERFORMANCE SUMMARY ---")
        
        if pb_results:
            print("\nPowerball:")
            print("Method                   RMSE       N")
            print("-" * 50)
            for method, data in pb_results.items():
                perf = data['performance']
                rmse = perf.get('rmse', 'N/A')
                n = perf.get('n_samples', 'N/A')
                if isinstance(rmse, float):
                    print(f"{method:24} {rmse:8.4f}   {n}")
                else:
                    print(f"{method:24} {rmse:>8}   {n}")
        
        if mm_results:
            print("\nMega Millions:")
            print("Method                   RMSE       N")
            print("-" * 50)
            for method, data in mm_results.items():
                perf = data['performance']
                rmse = perf.get('rmse', 'N/A')
                n = perf.get('n_samples', 'N/A')
                if isinstance(rmse, float):
                    print(f"{method:24} {rmse:8.4f}   {n}")
                else:
                    print(f"{method:24} {rmse:>8}   {n}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ensemble_builder.py <powerball_dir> <megamillions_dir> <outputs_dir>")
        sys.exit(1)
    
    builder = EnsembleBuilder(sys.argv[1], sys.argv[2], sys.argv[3])
    builder.run()
