#!/usr/bin/env python3
# =============================================================================
# Update Author Information
# =============================================================================
# This script updates the author information in all Python files.
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

# Define the new author block
AUTHOR_BLOCK = """# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
# Date:)',
            AUTHOR_BLOCK + '\n', 
            content,
            flags=re.DOTALL
        )
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        return True
    else:
        print(f"Skipping {file_path}: No author line found")
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
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                total_files += 1
                
                if update_file(file_path):
                    updated_files += 1
                    print(f"Updated: {file_path}")
    
    print(f"\nTotal Python files: {total_files}")
    print(f"Updated files: {updated_files}")

if __name__ == "__main__":
    main()
