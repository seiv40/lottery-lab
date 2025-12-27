"""
Generate VAE Universe - Module 7

"""

import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json

# add current directory to path for Module 7 imports
sys.path.insert(0, str(Path(__file__).parent))

# try to import from same directory, fall back to outputs directory
try:
    from module7_decoder import LotteryDecoder
    from module7_generation import UniverseGenerator, UniverseStatistics
except ImportError:
    # if not in same dir, try absolute Windows path
    sys.path.insert(0, r'C:\jackpotmath\lottery-lab\scripts')
    from module7_decoder import LotteryDecoder
    from module7_generation import UniverseGenerator, UniverseStatistics


# VAE ARCHITECTURE (Copied from vae_model.py)

class VAE(nn.Module):
    """
    Exact VAE architecture from Module 5.
    
    Architecture:
    - Encoder: input_dim -> 128 -> 64 -> latent_dim
    - Decoder: latent_dim -> 64 -> 128 -> input_dim
    """
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder: X -> z_mu, z_log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

        # decoder: z -> X_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization trick with stabilization."""
        log_var = torch.clamp(log_var, -20, 20)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode x to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        """Decode latent z to reconstructed x."""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass: encode, reparameterize, decode."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


# SCALER CLASS (minimal implementation)

class SimpleScaler:
    """Simple StandardScaler replacement if scaler file not found."""
    def __init__(self, mean=None, scale=None):
        self.mean_ = mean
        self.scale_ = scale
    
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # avoid division by zero
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def inverse_transform(self, X):
        """Inverse transform: convert standardized values back to original scale."""
        return X * self.scale_ + self.mean_


# PYTORCH VAE UNIVERSE GENERATOR

class PyTorchVAEUniverseGenerator(UniverseGenerator):
    """Generate universe using PyTorch VAE model."""
    
    def __init__(self,
                 lottery_type: str,
                 vae_model: nn.Module,
                 latent_dim: int,
                 scaler,
                 training_data: pd.DataFrame,
                 device: str = 'cpu'):
        super().__init__(lottery_type)
        
        self.vae_model = vae_model
        self.vae_model.eval()
        self.latent_dim = latent_dim
        self.scaler = scaler
        self.decoder = LotteryDecoder(lottery_type, training_data)
        self.device = device
        
        print(f"PyTorch VAE Universe Generator initialized")
        print(f"  Lottery: {lottery_type}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Device: {device}")
    
    def _decode_vae(self, z: np.ndarray) -> np.ndarray:
        """Decode latent code through VAE."""
        with torch.no_grad():
            z_tensor = torch.FloatTensor(z).to(self.device)
            
            if z_tensor.ndim == 1:
                z_tensor = z_tensor.unsqueeze(0)
            
            # decode
            decoded = self.vae_model.decode(z_tensor)
            
            features = decoded.cpu().numpy()
            
            if features.shape[0] == 1:
                features = features.squeeze(0)
            
            return features
    
    def generate(self, n_draws: int = 10000, seed: int = None, verbose: bool = True):
        """Generate universe using PyTorch VAE."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        if verbose:
            print(f"\nGenerating VAE Universe (PyTorch)")
            print(f"{'='*60}")
            print(f"Lottery: {self.lottery_type.upper()}")
            print(f"Draws: {n_draws:,}")
            if seed is not None:
                print(f"Seed: {seed}")
            print(f"{'='*60}\n")
        
        draws = []
        
        for i in range(n_draws):
            # sample from latent space N(0, I)
            z = np.random.randn(self.latent_dim)
            
            # decode through VAE
            features = self._decode_vae(z)
            
            # decode to lottery draw
            draw = self.decoder.decode_to_lottery_draw(features, self.scaler)
            
            # format as row
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
        """Print summary statistics."""
        print(f"\nVAE Universe Summary:")
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


# MAIN SCRIPT

def main(lottery='powerball'):
    """Generate VAE universe for specified lottery."""
    
    LOTTERY_CONFIG = {
        'powerball': {
            'model_path': r'C:\jackpotmath\lottery-lab\models\vae\powerball_vae_gen.pt',
            'scaler_path': r'C:\jackpotmath\lottery-lab\models\vae\powerball_vae_gen_scaler.pkl',
            'draws_path': r'C:\jackpotmath\lottery-lab\data\raw\powerball_draws.parquet',
            'output_path': r'C:\jackpotmath\lottery-lab\output\universe_vae_pb.parquet',
            'special_ball_name': 'powerball',
            'special_ball_max': 26
        },
        'megamillions': {
            'model_path': r'C:\jackpotmath\lottery-lab\models\vae\megamillions_vae_gen.pt',
            'scaler_path': r'C:\jackpotmath\lottery-lab\models\vae\megamillions_vae_gen_scaler.pkl',
            'draws_path': r'C:\jackpotmath\lottery-lab\data\raw\megamillions_draws.parquet',
            'output_path': r'C:\jackpotmath\lottery-lab\output\universe_vae_mm.parquet',
            'special_ball_name': 'megaball',
            'special_ball_max': 25
        }
    }
    
    config = LOTTERY_CONFIG[lottery]
    
    print("="*70)
    print(f"GENERATING VAE UNIVERSE - {lottery.upper()}")
    print("="*70)
    
    # CONFIGURATION (Paths from uploaded files)
    
    # model checkpoint
    MODEL_PATH = config['model_path']
    
    # raw draws data (same as training)
    DRAWS_PATH = config['draws_path']
    
    # output
    OUTPUT_PATH = config['output_path']
    
    # model hyperparameters (must match training)
    MODEL_KWARGS = {
        'input_dim': 6,        # 5 white balls + 1 special ball
        'latent_dim': 2,       # latent dimension
        'hidden_dim': 128      # hidden layer size
    }
    
    device = 'cpu'
    
    # LOAD DRAWS DATA
    
    print("\n[1/6] Loading draws data...")
    
    df = pd.read_parquet(DRAWS_PATH)
    print(f" Loaded {len(df):,} draws from {Path(DRAWS_PATH).name}")
    
    # extract ball columns (match training exactly)
    ball_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', config['special_ball_name']]
    
    print(f" Extracted {len(ball_cols)} ball columns: {ball_cols}")
    print(f"  Shape: {df[ball_cols].shape}")
    
    # verify all balls are within valid range
    for col in ball_cols[:5]:  # white balls
        assert df[col].between(1, 69 if lottery == 'powerball' else 70).all(), f"{col} out of range"
    assert df[ball_cols[5]].between(1, config['special_ball_max']).all(), f"{ball_cols[5]} out of range"
    
    print(f"  Range validation passed")
    
    # LOAD SCALER
    
    print("\n[2/6] Loading scaler...")
    
    scaler_path = config['scaler_path']
    
    # try loading saved scaler from training
    try:
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f" Loaded scaler: {Path(scaler_path).name}")
        print(f"  Mean: [{scaler.mean_[:3].round(2)}...]")
        print(f"  Scale: [{scaler.scale_[:3].round(2)}...]")
    except FileNotFoundError:
        # fallback: create scaler from training data
        print(f" Scaler not found, creating from training data...")
        
        # prepare ball data
        X = df[ball_cols].values.astype(np.float32)
        
        # split 70/15/15 (matching training)
        n = len(X)
        n_train = int(n * 0.70)
        X_train = X[:n_train]
        
        print(f" Training samples: {len(X_train):,}")
        
        # fit scaler on training data only
        scaler = SimpleScaler()
        scaler.fit(X_train)
        
        print(f" Scaler fitted")
        print(f"  Mean: [{scaler.mean_[:3].round(2)}...]")
        print(f"  Scale: [{scaler.scale_[:3].round(2)}...]")
    
    # LOAD VAE MODEL
    
    print("\n[3/6] Loading VAE model...")
    
    # load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(f" Loaded checkpoint from {Path(MODEL_PATH).name}")
    
    # initialize model
    model = VAE(**MODEL_KWARGS)
    
    # load weights
    if isinstance(checkpoint, dict):
        # state dict
        model.load_state_dict(checkpoint)
        print(f" Loaded state dict")
    else:
        # full model
        model = checkpoint
        print(f" Loaded model object")
    
    model.to(device)
    model.eval()
    
    print(f" Model ready")
    print(f"  Input dim: {MODEL_KWARGS['input_dim']}")
    print(f"  Latent dim: {MODEL_KWARGS['latent_dim']}")
    print(f"  Hidden dim: {MODEL_KWARGS['hidden_dim']}")
    
    # PREPARE DECODER DATA
    
    print("\n[4/6] Preparing decoder...")
    
    # load actual lottery draws for decoder
    DRAWS_PATH = config['draws_path']
    
    try:
        draws_df = pd.read_parquet(DRAWS_PATH)
        print(f" Loaded draw data: {len(draws_df)} draws from {DRAWS_PATH}")
        
        # verify columns
        required_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5', config['special_ball_name']]
        missing_cols = [c for c in required_cols if c not in draws_df.columns]
        
        if missing_cols:
            print(f" Missing columns: {missing_cols}")
            print(f"  Available columns: {list(draws_df.columns)}")
            raise ValueError(f"Draw data missing required columns: {missing_cols}")
        
        train_data = draws_df[required_cols].copy()
        print(f" Decoder data prepared: {len(train_data)} samples with all {len(required_cols)} ball columns")
        
    except FileNotFoundError:
        print(f" Draw data not found at: {DRAWS_PATH}")
        print(f"  Falling back to minimal training data...")
        # fallback to minimal data
        train_data = pd.DataFrame({
            'white_1': [10, 20, 30, 40, 50],
            'white_2': [15, 25, 35, 45, 55],
            'white_3': [20, 30, 40, 50, 60],
            'white_4': [25, 35, 45, 55, 65],
            'white_5': [30, 40, 50, 60, 69],
            config['special_ball_name']: [5, 10, 15, 20, config['special_ball_max']]
        })
        print(f" Using fallback data: {len(train_data)} samples")
    
    # GENERATE UNIVERSE
    
    print("\n[5/6] Generating universe...")
    print("This might take a few minutes...\n")
    
    vae_gen = PyTorchVAEUniverseGenerator(
        lottery_type=lottery,
        vae_model=model,
        latent_dim=MODEL_KWARGS['latent_dim'],
        scaler=scaler,
        training_data=train_data,
        device=device
    )
    
    universe_vae = vae_gen.generate(
        n_draws=10000,
        seed=42,
        verbose=True
    )
    
    # SAVE AND VALIDATE
    
    print("\n[6/6] Saving universe...")
    
    # ensure output directory exists
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vae_gen.save_universe(
        universe_vae,
        OUTPUT_PATH,
        metadata={
            'description': 'VAE-generated Powerball universe',
            'model_checkpoint': str(MODEL_PATH),
            'input_dim': MODEL_KWARGS['input_dim'],
            'latent_dim': MODEL_KWARGS['latent_dim'],
            'hidden_dim': MODEL_KWARGS['hidden_dim'],
            'n_draws': len(universe_vae),
            'seed': 42
        }
    )
    
    print(f"\n Universe saved to: {OUTPUT_PATH}")
    
    # quick validation
    print("\n--- VALIDATION ---")
    
    stats = UniverseStatistics(lottery)
    universe_stats = stats.compute_statistics(universe_vae)
    stats.print_statistics(universe_stats)
    
    # check for common issues
    white_cols = ['white_1', 'white_2', 'white_3', 'white_4', 'white_5']
    special_col = config['special_ball_name']
    special_max = config['special_ball_max']
    issues = 0
    
    for i, row in universe_vae.head(100).iterrows():
        white_balls = row[white_cols].values
        
        # check uniqueness
        if len(set(white_balls)) != 5:
            print(f" Draw {i} has duplicate balls")
            issues += 1
        
        # check sorted
        if list(white_balls) != sorted(white_balls):
            print(f" Draw {i} not sorted")
            issues += 1
        
        # check range
        if not all(1 <= b <= 69 for b in white_balls):
            print(f" Draw {i} has out-of-range white balls")
            issues += 1
        
        # check powerball/megaball
        if not (1 <= row[special_col] <= special_max):
            print(f" Draw {i} has invalid {special_col}")
            issues += 1
    
    if issues == 0:
        print("\n All validation checks passed!")
    else:
        print(f"\n Found {issues} issues in first 100 draws")
    
    print("\n--- VAE UNIVERSE GENERATION COMPLETE! ---")
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"\nNext steps:")
    print(f"  1. Generate Mega Millions VAE universe")
    print(f"  2. Generate Flow universes")
    print(f"  3. Generate Transformer universes")
    print(f"  4. Run complete analysis (module7_complete_analysis.py)")


if __name__ == "__main__":
    import sys
    
    # get lottery from command line, default to powerball
    if len(sys.argv) > 1:
        lottery = sys.argv[1].lower()
        if lottery not in ['powerball', 'megamillions']:
            print(f"Error: Unknown lottery '{lottery}'. Use 'powerball' or 'megamillions'")
            sys.exit(1)
    else:
        lottery = 'powerball'
    
    main(lottery)
