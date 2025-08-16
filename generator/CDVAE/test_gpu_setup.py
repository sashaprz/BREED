#!/usr/bin/env python3
"""
Test script to verify GPU setup and PyTorch Lightning configuration
for CDVAE training troubleshooting.
"""

import sys
import torch
import pytorch_lightning as pl
from packaging import version

def test_basic_gpu_setup():
    """Test basic PyTorch GPU functionality"""
    print("=" * 60)
    print("BASIC GPU SETUP TEST")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute capability: {props.major}.{props.minor}")
        
        # Test basic GPU operations
        try:
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print(f"✓ Basic GPU tensor operations working")
            print(f"  - Test tensor shape: {z.shape}")
            print(f"  - Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
    else:
        print("No CUDA GPUs available - will use CPU")
    
    return cuda_available

def test_pytorch_lightning_config():
    """Test PyTorch Lightning trainer configuration"""
    print("\n" + "=" * 60)
    print("PYTORCH LIGHTNING CONFIGURATION TEST")
    print("=" * 60)
    
    # Check PyTorch Lightning version compatibility
    pl_version = version.parse(pl.__version__)
    print(f"PyTorch Lightning version: {pl.__version__}")
    
    if pl_version >= version.parse("1.7.0"):
        print("✓ Using modern PyTorch Lightning (>=1.7.0)")
        print("  - Should use 'accelerator' and 'devices' instead of 'gpus'")
        modern_pl = True
    else:
        print("⚠ Using older PyTorch Lightning (<1.7.0)")
        print("  - Can use 'gpus' parameter")
        modern_pl = False
    
    # Test trainer instantiation with different configurations
    cuda_available = torch.cuda.is_available()
    
    print("\nTesting trainer configurations:")
    
    # Test 1: CPU trainer
    try:
        trainer_cpu = pl.Trainer(
            accelerator="cpu",
            max_epochs=1,
            enable_progress_bar=False,
            logger=False
        )
        print("✓ CPU trainer configuration works")
    except Exception as e:
        print(f"✗ CPU trainer failed: {e}")
    
    # Test 2: GPU trainer (if available)
    if cuda_available:
        try:
            if modern_pl:
                trainer_gpu = pl.Trainer(
                    accelerator="gpu",
                    devices=1,
                    max_epochs=1,
                    enable_progress_bar=False,
                    logger=False
                )
                print("✓ Modern GPU trainer configuration works (accelerator='gpu', devices=1)")
            else:
                trainer_gpu = pl.Trainer(
                    gpus=1,
                    max_epochs=1,
                    enable_progress_bar=False,
                    logger=False
                )
                print("✓ Legacy GPU trainer configuration works (gpus=1)")
        except Exception as e:
            print(f"✗ GPU trainer failed: {e}")
            
        # Test 3: Try the old configuration that's causing issues
        try:
            trainer_old = pl.Trainer(
                gpus=1,
                max_epochs=1,
                enable_progress_bar=False,
                logger=False
            )
            print("⚠ Old 'gpus=1' configuration still works (might be deprecated)")
        except Exception as e:
            print(f"✗ Old 'gpus=1' configuration fails: {e}")
            print("  - This is likely why your training is hanging!")
    
    return modern_pl, cuda_available

def test_simple_lightning_module():
    """Test a simple PyTorch Lightning module"""
    print("\n" + "=" * 60)
    print("SIMPLE LIGHTNING MODULE TEST")
    print("=" * 60)
    
    class SimpleModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            return self.layer(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
    
    # Create simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            x = torch.randn(10)
            y = torch.randn(1)
            return x, y
    
    try:
        model = SimpleModel()
        dataset = SimpleDataset(20)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Test with CPU
        trainer = pl.Trainer(
            accelerator="cpu",
            max_epochs=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False
        )
        trainer.fit(model, dataloader)
        print("✓ Simple Lightning module works on CPU")
        
        # Test with GPU if available
        if torch.cuda.is_available():
            model_gpu = SimpleModel()
            trainer_gpu = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=1,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False
            )
            trainer_gpu.fit(model_gpu, dataloader)
            print("✓ Simple Lightning module works on GPU")
            
    except Exception as e:
        print(f"✗ Simple Lightning module test failed: {e}")

def generate_recommendations():
    """Generate recommendations based on test results"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR CDVAE TRAINING")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    pl_version = version.parse(pl.__version__)
    modern_pl = pl_version >= version.parse("1.7.0")
    
    print("Based on the test results, here are the recommended fixes:")
    print()
    
    if modern_pl:
        print("1. UPDATE TRAINER CONFIGURATION:")
        print("   Replace in conf/train/default.yaml:")
        print("   OLD: gpus: 1")
        print("   NEW: accelerator: gpu")
        print("        devices: 1")
        print()
    
    print("2. ADD MISSING PROGRESS BAR CONFIG:")
    print("   Add to conf/logging/default.yaml:")
    print("   enable_progress_bar: true")
    print()
    
    if cuda_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8:
            print("3. REDUCE BATCH SIZE (Low GPU Memory):")
            print("   Your GPU has limited memory. Consider:")
            print("   batch_size:")
            print("     train: 32")
            print("     val: 32")
            print("     test: 32")
            print()
    
    print("4. UPDATED CONFIGURATION EXAMPLE:")
    if modern_pl and cuda_available:
        print("""
   pl_trainer:
     fast_dev_run: False
     accelerator: gpu
     devices: 1
     precision: 32
     max_epochs: ${data.train_max_epochs}
     accumulate_grad_batches: 1
     num_sanity_val_steps: 2
     gradient_clip_val: 0.5
     gradient_clip_algorithm: value
     profiler: simple
        """)
    elif cuda_available:
        print("""
   pl_trainer:
     fast_dev_run: False
     gpus: 1  # Keep this for older PyTorch Lightning
     precision: 32
     max_epochs: ${data.train_max_epochs}
     accumulate_grad_batches: 1
     num_sanity_val_steps: 2
     gradient_clip_val: 0.5
     gradient_clip_algorithm: value
     profiler: simple
        """)
    else:
        print("""
   pl_trainer:
     fast_dev_run: False
     accelerator: cpu
     precision: 32
     max_epochs: ${data.train_max_epochs}
     accumulate_grad_batches: 1
     num_sanity_val_steps: 2
     gradient_clip_val: 0.5
     gradient_clip_algorithm: value
     profiler: simple
        """)

def main():
    """Run all tests and generate recommendations"""
    print("CDVAE GPU Setup and PyTorch Lightning Configuration Test")
    print("=" * 60)
    
    try:
        # Run tests
        cuda_available = test_basic_gpu_setup()
        modern_pl, _ = test_pytorch_lightning_config()
        test_simple_lightning_module()
        
        # Generate recommendations
        generate_recommendations()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        print("Run this script to diagnose your CDVAE training issues.")
        print("Follow the recommendations above to fix the configuration.")
        
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()