#!/usr/bin/env python3
"""
YAMNet Fine-tuning Pipeline Runner
Combines data preparation and training steps
"""

import argparse
import sys
import os

def run_data_preparation():
    """Run data preparation step"""
    print("🔄 Starting Data Preparation...")
    exit_code = os.system(f"{sys.executable} prepare_yamnet_speech_nonspeech.py")
    if exit_code != 0:
        print("❌ Data preparation failed!")
        return False
    print("✅ Data preparation completed")
    return True

def run_training():
    """Run model training step"""
    print("🔄 Starting Model Training...")
    exit_code = os.system(f"{sys.executable} finetune_yamnet_speech_nonspeech.py")
    if exit_code != 0:
        print("❌ Training failed!")
        return False
    print("✅ Training completed")
    return True

def main():
    parser = argparse.ArgumentParser(description="YAMNet Fine-tuning Pipeline")
    parser.add_argument("--step", choices=["data", "train", "all"], default="all",
                       help="Pipeline step to run")
    
    args = parser.parse_args()
    
    if args.step in ["data", "all"]:
        if not run_data_preparation():
            sys.exit(1)
    
    if args.step in ["train", "all"]:
        if not run_training():
            sys.exit(1)
    
    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()