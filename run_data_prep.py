#!/usr/bin/env python3
"""
Enhanced data preparation script with multiple options.
Usage examples:
  python run_data_prep.py --basic                           # Basic data preparation
  python run_data_prep.py --augmented                       # With data augmentation
  python run_data_prep.py --validate                        # Only validate existing data
  python run_data_prep.py --small --speech 400 --nonspeech 400  # Small dataset
"""

import subprocess
import sys
import argparse
import os

def run_data_preparation(config_name, **kwargs):
    """Run data preparation with predefined or custom configuration."""
    
    # Base command
    cmd = [sys.executable, "prepare_yamnet_speech_nonspeech.py"]
    
    # Predefined configurations
    configs = {
        "basic": {
            "speech_samples": 800,
            "nonspeech_samples": 800,
            "output_dir": "yamnet_dataset"
        },
        "small": {
            "speech_samples": 400,
            "nonspeech_samples": 400,
            "output_dir": "yamnet_dataset_small"
        },
        "large": {
            "speech_samples": 1200,
            "nonspeech_samples": 1200,
            "output_dir": "yamnet_dataset_large"
        },
        "augmented": {
            "speech_samples": 800,
            "nonspeech_samples": 800,
            "augment": True,
            "output_dir": "yamnet_dataset_augmented"
        },
        "validate": {
            "validate_only": True,
            "speech_samples": 800,
            "nonspeech_samples": 800
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
        if isinstance(value, bool) and value:
            cmd.append(f"--{key.replace('_', '-')}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run data preparation
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        if config.get('validate_only'):
            print("Data validation completed! ✅")
        else:
            print("Data preparation completed successfully! 🎉")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nData preparation failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nData preparation interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run data preparation with predefined configurations")
    
    # Predefined configurations
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--basic", action="store_true", help="Basic data preparation (800 samples each)")
    group.add_argument("--small", action="store_true", help="Small dataset (400 samples each)")
    group.add_argument("--large", action="store_true", help="Large dataset (1200 samples each)")
    group.add_argument("--augmented", action="store_true", help="Basic dataset with augmentation")
    group.add_argument("--validate", action="store_true", help="Only validate data quality")
    group.add_argument("--custom", action="store_true", help="Custom configuration")
    
    # Custom configuration options
    parser.add_argument("--speech", "--speech-samples", type=int, dest="speech_samples", 
                       help="Number of speech samples")
    parser.add_argument("--nonspeech", "--nonspeech-samples", type=int, dest="nonspeech_samples",
                       help="Number of non-speech samples")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing data")
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.basic:
        success = run_data_preparation("basic")
    elif args.small:
        success = run_data_preparation("small")
    elif args.large:
        success = run_data_preparation("large")
    elif args.augmented:
        success = run_data_preparation("augmented")
    elif args.validate:
        success = run_data_preparation("validate")
    elif args.custom:
        # Build custom config from args
        custom_config = {}
        for key, value in vars(args).items():
            if value is not None and key not in ["basic", "small", "large", "augmented", "validate", "custom"]:
                custom_config[key] = value
        
        if not custom_config:
            print("No custom configuration provided. Using basic configuration.")
            success = run_data_preparation("basic")
        else:
            success = run_data_preparation("custom", **custom_config)
    else:
        # Default to basic if nothing specified
        print("No configuration specified. Using basic configuration.")
        success = run_data_preparation("basic")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())