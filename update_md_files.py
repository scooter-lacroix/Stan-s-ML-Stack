#!/usr/bin/env python3
# =============================================================================
# Update Markdown Files
# =============================================================================
# This script updates the author information in all Markdown files.
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

# Define the author block for markdown files
AUTHOR_BLOCK = """
## Author

**Stanley Chisango (Scooter Lacroix)**

- Email: scooterlacroix@gmail.com
- GitHub: [scooter-lacroix](https://github.com/scooter-lacroix)
- X: [@scooter_lacroix](https://x.com/scooter_lacroix)
- Patreon: [ScooterLacroix](https://patreon.com/ScooterLacroix)

> If this code saved you time, consider buying me a coffee! ☕
> 
> "Code is like humor. When you have to explain it, it's bad!" - Cory House

"""

def update_md_file(file_path):
    """Update author information in a markdown file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if the file already has an author section
        if re.search(r'## Author', content):
            # Replace existing author section
            updated_content = re.sub(
                r'## Author.*?(?=##|\Z)',
                AUTHOR_BLOCK,
                content,
                flags=re.DOTALL
            )
        else:
            # Add author section before the license section if it exists
            if re.search(r'## License', content):
                updated_content = re.sub(
                    r'(## License)',
                    AUTHOR_BLOCK + r'\1',
                    content
                )
            else:
                # Add author section at the end of the file
                updated_content = content.rstrip() + "\n\n" + AUTHOR_BLOCK
        
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
            if filename.endswith('.md'):
                file_path = os.path.join(dirpath, filename)
                total_files += 1
                
                if update_md_file(file_path):
                    updated_files += 1
    
    print(f"\nTotal Markdown files: {total_files}")
    print(f"Updated files: {updated_files}")

if __name__ == "__main__":
    main()
