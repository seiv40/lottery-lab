"""
Generate Flow Universe - Module 7

"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent))

try:
    from module7_generation import UniverseGenerator, UniverseStatistics
except ImportError:
    sys.path.insert(0, r'C:\jackpotmath\lottery-lab\scripts')
    from module7_generation import UniverseGenerator, UniverseStatistics


# FLOW ARCHITECTURE

class MaskedAffineFlow(nn.Module):
    def __init__(self, dim, hidden_dim=128, mask_type='even'):
        super().__init__()
        self.dim = dim
        self.mask = self.build_mask(mask_type)
        
        self.s_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim)
        )
        self.t_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, dim)
        )
        
        nn.init.zeros_(self.s_net[-1].weight.data)
        nn.init.zeros_(self.s_net[-1].bias.data)

    def build_mask(self, mask_type):
        mask = torch.zeros(self.dim)
        if mask_type == 'even':
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        return mask.float()

    def forward(self, x):
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)
        x_masked = x * self.mask
        s_raw = self.s_net(x_masked)
        s = torch.tanh(s_raw) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        z = (x * torch.exp(s) + t) * (1 - self.mask) + x_masked
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        if self.mask.device != z.device:
            self.mask = self.mask.to(z.device)
        z_masked = z * self.mask
        s_raw = self.s_net(z_masked)
        s = torch.tanh(s_raw) * (1 - self.mask)
        t = self.t_net(z_masked) * (1 - self.mask)
        x = ((z - t) * torch.exp(-s)) * (1 - self.mask) + z_masked
        log_det = -s.sum(dim=-1)
        return x, log_det


class MAF(nn.Module):
    def __init__(self, dim, n_flows=5, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.base_dist = torch.distributions.Normal(torch.zeros(dim), torch.ones(dim))
        flows = []
        for i in range(n_flows):
            mask_type = 'even' if i % 2 == 0 else 'odd'
            flows.append(MaskedAffineFlow(dim, hidden_dim, mask_type))
        self.flows = nn.Sequential(*flows)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.base_dist = torch.distributions.Normal(
            torch.zeros(self.dim).to(*args, **kwargs),
            torch.ones(self.dim).to(*args, **kwargs)
        )
        return self

    def sample(self, n_samples=1):
        z = self.base_dist.sample((n_samples,))
        x = z
        for flow in reversed(self.flows):
            x, _ = flow.inverse(x)
        return x


# DECODER

class BallSpaceDecoder:
    def __init__(self, lottery_type: str):
        self.lottery_type = lottery_type.lower()
        
        if self.lottery_type == 'powerball':
            self.white_range = range(1, 70)
            self.special_range = range(1, 27)
            self.n_white = 5
            self.special_name = 'powerball'
        elif self.lottery_type == 'megamillions':
            self.white_range = range(1, 71)
            self.special_range = range(1, 26)
            self.n_white = 5
            self.special_name = 'megaball'
    
    def decode_to_lottery_draw(self, balls_continuous: np.ndarray, scaler) -> dict:
        if balls_continuous.ndim == 1:
            balls_continuous = balls_continuous.reshape(1, -1)
        
        balls_unscaled = scaler.inverse_transform(balls_continuous).flatten()
        balls_int = np.round(balls_unscaled).astype(int)
        
        white_balls = balls_int[:self.n_white]
        special_ball = balls_int[self.n_white]
        
        white_balls = np.clip(white_balls, min(self.white_range), max(self.white_range))
        white_balls = self._enforce_uniqueness(white_balls, self.white_range)
        white_balls = sorted(white_balls)
        
        special_ball = np.clip(special_ball, min(self.special_range), max(self.special_range))
        
        return {
            'white_balls': white_balls,
            self.special_name: int(special_ball)
        }
    
    def _enforce_uniqueness(self, balls: np.ndarray, valid_range: range) -> list:
        balls = list(balls)
        for _ in range(100):
            if len(set(balls)) == len(balls):
                return balls
            seen = set()
            duplicates = []
            for i, ball in enumerate(balls):
                if ball in seen:
                    duplicates.append(i)
                seen.add(ball)
            used = set(balls)
            available = [b for b in valid_range if b not in used]
            if not available:
                return sorted(np.random.choice(list(valid_range), len(balls), replace=False).tolist())
            for dup_idx in duplicates:
                nearest = min(available, key=lambda x: abs(x - balls[dup_idx]))
                balls[dup_idx] = nearest
                available.remove(nearest)
        return sorted(np.random.choice(list(valid_range), len(balls), replace=False).tolist())


# UNIVERSE GENERATOR

class FlowUniverseGenerator(UniverseGenerator):
    def __init__(self, lottery_type: str, flow_model: nn.Module, scaler, device: str = 'cpu'):
        super().__init__(lottery_type)
        self.flow_model = flow_model
        self.flow_model.eval()
        self.scaler = scaler
        self.decoder = BallSpaceDecoder(lottery_type)
        self.device = device
        
        print(f"Flow Universe Generator initialized")
        print(f"  Lottery: {lottery_type}")
        print(f"  Device: {device}")
    
    def generate(self, n_draws: int = 10000, seed: int = None, verbose: bool = True):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        if verbose:
            print(f"\nGenerating Flow Universe")
            print(f"{'-'*60}")
            print(f"Lottery: {self.lottery_type.upper()}")
            print(f"Draws: {n_draws:,}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'-'*60}\n")
        
        draws = []
        
        with torch.no_grad():
            for i in range(n_draws):
                # sample from flow
                sample = self.flow_model.sample(1).cpu().numpy().squeeze()
                
                # decode to lottery draw
                draw = self.decoder.decode_to_lottery_draw(sample, self.scaler)
                
                draw_row = {
                    'draw_id': i + 1,
                    'white_1': draw['white_balls'][0],
                    'white_2': draw['white_balls'][1],
                    'white_3': draw['white_balls'][2],
                    'white_4': draw['white_balls'][3],
                    'white_5': draw['white_balls'][4],
                    self.special_name: draw[self.special_name]
                }
                draws.append(draw_row)
                
                if verbose and (i + 1) % 1000 == 0:
                    print(f"  Generated {i + 1:,} draws...")
        
        universe = pd.DataFrame(draws)
        
        if verbose:
            print(f"\n Generated {len(universe):,} draws")
            self._print_universe_summary(universe)
        
        return universe
    
    def _print_universe_summary(self, universe: pd.DataFrame):
        print(f"\nFlow Universe Summary:")
        print(f"-" * 40)
        
        white_cols = [f'white_{i+1}' for i in range(self.n_white)]
        all_white_balls = universe[white_cols].values.flatten()
        
        print(f"White balls:")
        print(f"  Mean: {all_white_balls.mean():.2f}")
        print(f"  Std: {all_white_balls.std():.2f}")
        print(f"  Range: {all_white_balls.min():.0f} - {all_white_balls.max():.0f}")
        
        special_balls = universe[self.special_name].values
        print(f"\n{self.special_name.capitalize()}:")
        print(f"  Mean: {special_balls.mean():.2f}")
        print(f"  Unique values: {len(np.unique(special_balls))}")
        print(f"-" * 40)


# MAIN

def main(lottery: str):
    lottery = lottery.lower()
    if lottery not in ['powerball', 'megamillions']:
        raise ValueError(f"Unknown lottery: {lottery}")
    
    MODELS_DIR = Path(r'C:\jackpotmath\lottery-lab\models\flow')
    OUTPUT_DIR = Path(r'C:\jackpotmath\lottery-lab\output')
    
    suffix = '_pb' if lottery == 'powerball' else '_mm'
    
    MODEL_PATH = MODELS_DIR / f"{lottery}_flow_gen.pt"
    SCALER_PATH = MODELS_DIR / f"{lottery}_flow_gen_scaler.pkl"
    OUTPUT_PATH = OUTPUT_DIR / f"universe_flow{suffix}.parquet"
    
    MODEL_KWARGS = {
        'dim': 6,
        'n_flows': 5,
        'hidden_dim': 128
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n--- GENERATING FLOW UNIVERSE - {lottery.upper()} ---")
    
    # load model
    print("\n[1/3] Loading Flow model...")
    
    if not MODEL_PATH.exists():
        print(f"\n Model not found: {MODEL_PATH}")
        print(f"\nTrain first:")
        print(f"  python flow_generative_model.py --lottery {lottery} --verbose")
        return
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(f" Loaded checkpoint")
    
    model = MAF(**MODEL_KWARGS)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f" Model ready")
    
    # load scaler
    print("\n[2/3] Loading scaler...")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f" Scaler loaded")
    
    # generate
    print("\n[3/3] Generating universe...\n")
    
    flow_gen = FlowUniverseGenerator(
        lottery_type=lottery,
        flow_model=model,
        scaler=scaler,
        device=device
    )
    
    universe_flow = flow_gen.generate(n_draws=10000, seed=42, verbose=True)
    
    # save
    print("\n[4/4] Saving...")
    
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    flow_gen.save_universe(
        universe_flow,
        str(OUTPUT_PATH),
        metadata={
            'description': f'Flow-generated {lottery} universe',
            'model_checkpoint': str(MODEL_PATH),
            'n_flows': MODEL_KWARGS['n_flows'],
            'n_draws': len(universe_flow),
            'seed': 42,
            'model_type': 'normalizing_flow',
            'trained_on': 'ball_positions'
        }
    )
    
    print(f"\n Saved: {OUTPUT_PATH}")
    
    # validate
    print("\n--- VALIDATION ---")
    
    stats = UniverseStatistics(lottery)
    universe_stats = stats.compute_statistics(universe_flow)
    stats.print_statistics(universe_stats)
    
    # quick checks
    white_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']
    special_col = flow_gen.special_name
    issues = 0
    
    for i, row in universe_flow.head(100).iterrows():
        white_balls = row[white_cols].values
        
        if len(set(white_balls)) != 5:
            print(f" Draw {i} has duplicate balls")
            issues += 1
        
        if list(white_balls) != sorted(white_balls):
            print(f" Draw {i} not sorted")
            issues += 1
    
    if issues == 0:
        print("\n All validation checks passed!")
    else:
        print(f"\n Found {issues} issues in first 100 draws")
    
    print("\n--- FLOW UNIVERSE COMPLETE! ---")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        lottery = sys.argv[1].lower()
        if lottery not in ['powerball', 'megamillions']:
            print(f"Error: Unknown lottery '{lottery}'")
            sys.exit(1)
    else:
        lottery = 'powerball'
    
    main(lottery)
