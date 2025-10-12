#!/usr/bin/env python3
"""
Convenient script to run YAMNet training with different configurations.
Usage examples:
  python run_training.py --basic                    # Basic training
  python run_training.py --advanced                 # Advanced training with all features
  python run_training.py --quick                    # Quick training (fewer epochs)
  python run_training.py --custom --lr 0.001 --epochs 50  # Custom configuration
"""

import subprocess
import sys
import argparse
import os

def run_training(config_name, **kwargs):
    """Run training with predefined or custom configuration."""
    
    # Base command
    cmd = [sys.executable, "finetune_yamnet_speech_nonspeech.py"]
    
    # Predefined configurations
    configs = {
        "basic": {
            "batch_size": 32,
            "epochs": 20,
            "lr": 0.001,
            "dropout": 0.3,
            "patience": 5,
            "model_name": "yamnet_basic"
        },
        "advanced": {
            "batch_size": 32,
            "epochs": 50,
            "lr": 0.001,
            "dropout": 0.3,
            "patience": 7,
            "use_scheduler": True,
            "use_tensorboard": True,
            "model_name": "yamnet_advanced"
        },
        "quick": {
            "batch_size": 64,
            "epochs": 10,
            "lr": 0.002,
            "dropout": 0.2,
            "patience": 3,
            "model_name": "yamnet_quick"
        },
        "production": {
            "batch_size": 32,
            "epochs": 100,
            "lr": 0.0005,
            "dropout": 0.4,
            "patience": 10,
            "use_scheduler": True,
            "use_tensorboard": True,
            "model_name": "yamnet_production"
        }
    }
    
    # Get configuration
    if config_name in configs:
        config = configs[config_name]
        print(f"Using predefined configuration: {config_name}")
        print(f"Configuration: {config}")
    else:
        config = kwargs
        print(f"Using custom configuration: {config}")
    
    # Build command arguments
    for key, value in config.items():
        if key.startswith("use_") and value:
            cmd.append(f"--{key.replace('_', '-')}")
        elif not key.startswith("use_"):
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Training completed successfully! 🎉")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run YAMNet training with predefined configurations")
    
    # Predefined configurations
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--basic", action="store_true", help="Basic training configuration")
    group.add_argument("--advanced", action="store_true", help="Advanced training with scheduler and TensorBoard")
    group.add_argument("--quick", action="store_true", help="Quick training (fewer epochs)")
    group.add_argument("--production", action="store_true", help="Production-ready training (long)")
    group.add_argument("--custom", action="store_true", help="Custom configuration")
    
    # Custom configuration options
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--model-name", type=str, help="Model name prefix")
    parser.add_argument("--use-scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard")
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.basic:
        success = run_training("basic")
    elif args.advanced:
        success = run_training("advanced")
    elif args.quick:
        success = run_training("quick")
    elif args.production:
        success = run_training("production")
    elif args.custom:
        # Build custom config from args
        custom_config = {}
        for key, value in vars(args).items():
            if value is not None and key not in ["basic", "advanced", "quick", "production", "custom"]:
                custom_config[key] = value
        
        if not custom_config:
            print("No custom configuration provided. Using default values.")
            custom_config = {"model_name": "yamnet_custom"}
        
        success = run_training("custom", **custom_config)
    else:
        # Default to basic if nothing specified
        print("No configuration specified. Using basic configuration.")
        success = run_training("basic")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())