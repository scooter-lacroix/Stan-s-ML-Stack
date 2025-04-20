#!/usr/bin/env python3
# =============================================================================
# Update Text Files
# =============================================================================
# This script updates the author information in text files.
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
import re
import sys

# Define the author block for text files
AUTHOR_BLOCK = """
Author: Stanley Chisango (Scooter Lacroix)
Email: scooterlacroix@gmail.com
GitHub: https://github.com/scooter-lacroix
X: https://x.com/scooter_lacroix
Patreon: https://patreon.com/ScooterLacroix

If this code saved you time, consider buying me a coffee! ☕
"Code is like humor. When you have to explain it, it's bad!" - Cory House

"""

def update_txt_file(file_path):
    """Update author information in a text file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file is a banner or ASCII art
        if "amd_ml_stack_banner.txt" in file_path:
            # Add author information at the end of the file
            updated_content = content.rstrip() + "\n\n" + AUTHOR_BLOCK
        else:
            # Check if the file already has author information
            if "Author:" in content:
                # Skip the file
                print(f"Skipping {file_path}: Already has author information")
                return False
            
            # Add author information at the beginning of the file
            updated_content = AUTHOR_BLOCK + content
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated: {file_path}")
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function."""
    # Get the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Count files
    total_files = 0
    updated_files = 0
    
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                total_files += 1
                
                if update_txt_file(file_path):
                    updated_files += 1
    
    print(f"\nTotal text files: {total_files}")
    print(f"Updated files: {updated_files}")

if __name__ == "__main__":
    main()
