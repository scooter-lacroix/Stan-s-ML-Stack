#!/usr/bin/env python3
# =============================================================================
# Update Remaining Files
# =============================================================================
# This script updates the author information in files that were skipped.
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
"""

# List of files to update
FILES_TO_UPDATE = [
    "/home/stan/Desktop/Stans_MLStack/core/flash_attention/flash_attention_amd.py",
    "/home/stan/Desktop/Stans_MLStack/core/flash_attention/__init__.py",
    "/home/stan/Desktop/Stans_MLStack/core/migraphx/setup.py",
    "/home/stan/Desktop/Stans_MLStack/core/migraphx/migraphx/__init__.py"
]

def update_file(file_path):
    """Update author information in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file has a shebang line
        if content.startswith('#!/'):
            # Add author block after the shebang line
            lines = content.split('\n')
            updated_content = lines[0] + '\n' + AUTHOR_BLOCK + '\n' + '\n'.join(lines[1:])
        else:
            # Add author block at the beginning of the file
            updated_content = AUTHOR_BLOCK + '\n' + content
        
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated: {file_path}")
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function."""
    # Count files
    total_files = len(FILES_TO_UPDATE)
    updated_files = 0
    
    # Update each file
    for file_path in FILES_TO_UPDATE:
        if os.path.exists(file_path):
            if update_file(file_path):
                updated_files += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nTotal files to update: {total_files}")
    print(f"Updated files: {updated_files}")

if __name__ == "__main__":
    main()
