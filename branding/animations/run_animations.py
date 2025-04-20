#!/usr/bin/env python3
# =============================================================================
# Run Animations
# =============================================================================
# This script runs all the animations for the ML Stack branding.
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date: 2023-04-19
# =============================================================================

import os
import subprocess
import sys

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the animations to run
animations = [
    "ml_stack_logo.py",
    "ml_stack_architecture.py",
    "patreon_tiers.py",
    "github_repo.py"
]

def run_animation(animation_file):
    """Run a Manim animation."""
    print(f"Running animation: {animation_file}")
    
    # Construct the command
    command = [
        "manim",
        "-pqh",  # Preview, medium quality, HD resolution
        animation_file,
        "MLStackLogo" if "logo" in animation_file else
        "MLStackArchitecture" if "architecture" in animation_file else
        "PatreonTiers" if "patreon" in animation_file else
        "GitHubRepo"
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True, cwd=script_dir)
        print(f"Animation {animation_file} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running animation {animation_file}: {e}")
        return False

def main():
    """Main function."""
    # Check if Manim is installed
    try:
        subprocess.run(["manim", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Manim is not installed or not in the PATH")
        print("Please install Manim first: pip install manim")
        return
    
    # Run each animation
    success_count = 0
    for animation in animations:
        animation_path = os.path.join(script_dir, animation)
        if os.path.exists(animation_path):
            if run_animation(animation_path):
                success_count += 1
        else:
            print(f"Animation file not found: {animation_path}")
    
    print(f"\nCompleted {success_count} out of {len(animations)} animations")

if __name__ == "__main__":
    main()
