#!/bin/bash
#
# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House
#
# =============================================================================
# ML Stack Git Preparation Script
# =============================================================================
# This script prepares the ML Stack for Git by cleaning up temporary files,
# organizing the directory structure, and creating necessary Git files.
# =============================================================================

# ASCII Art Banner
cat << "EOF"
  ██████╗████████╗ █████╗ ███╗   ██╗███████╗    ███╗   ███╗██╗         ███████╗████████╗ █████╗  ██████╗██╗  ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔════╝    ████╗ ████║██║         ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██║ ██╔╝
 ╚█████╗    ██║   ███████║██╔██╗ ██║███████╗    ██╔████╔██║██║         ███████╗   ██║   ███████║██║     █████╔╝ 
  ╚═══██╗   ██║   ██╔══██║██║╚██╗██║╚════██║    ██║╚██╔╝██║██║         ╚════██║   ██║   ██╔══██║██║     ██╔═██╗ 
 ██████╔╝   ██║   ██║  ██║██║ ╚████║███████║    ██║ ╚═╝ ██║███████╗    ███████║   ██║   ██║  ██║╚██████╗██║  ██╗
 ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                                                                                                                 
                                ML Stack Git Preparation Script
EOF
echo

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
BLINK='\033[5m'
REVERSE='\033[7m'
RESET='\033[0m'

# Function definitions
print_header() {
    echo -e "${CYAN}${BOLD}=== $1 ===${RESET}"
    echo
}

print_section() {
    echo -e "${BLUE}${BOLD}>>> $1${RESET}"
}

print_step() {
    echo -e "${MAGENTA}>> $1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

# Function to clean up temporary files
cleanup_temp_files() {
    print_section "Cleaning up temporary files"
    
    # Find and remove Python cache files
    print_step "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    
    # Find and remove temporary files
    print_step "Removing temporary files..."
    find . -type f -name "*.tmp" -delete
    find . -type f -name "*.temp" -delete
    find . -type f -name "*.swp" -delete
    find . -type f -name "*.swo" -delete
    find . -type f -name "*~" -delete
    
    # Find and remove build artifacts
    print_step "Removing build artifacts..."
    find . -type d -name "build" -exec rm -rf {} +
    find . -type d -name "dist" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    
    print_success "Temporary files cleaned up"
}

# Function to organize directory structure
organize_directory_structure() {
    print_section "Organizing directory structure"
    
    # Create necessary directories
    print_step "Creating directory structure..."
    mkdir -p docs/images
    mkdir -p docs/guides
    mkdir -p scripts
    mkdir -p logs
    mkdir -p data
    mkdir -p models
    mkdir -p benchmark_results
    mkdir -p test_results
    
    # Move files to appropriate directories
    print_step "Moving files to appropriate directories..."
    
    # Move documentation files
    find . -maxdepth 1 -type f -name "*.md" ! -name "README.md" -exec mv {} docs/ \;
    
    # Move script files
    find . -maxdepth 1 -type f -name "*.sh" -exec mv {} scripts/ \;
    find . -maxdepth 1 -type f -name "*.py" -exec mv {} scripts/ \;
    
    print_success "Directory structure organized"
}

# Function to create Git files
create_git_files() {
    print_section "Creating Git files"
    
    # Create .gitattributes
    print_step "Creating .gitattributes..."
    cat > .gitattributes << 'EOF'
# Auto detect text files and perform LF normalization
* text=auto

# Documents
*.md text
*.txt text
*.pdf binary

# Scripts
*.sh text eol=lf
*.py text eol=lf

# Data
*.json text
*.yaml text
*.yml text
*.csv text
*.tsv text

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.gz binary
*.zip binary
*.7z binary
*.ttf binary
*.eot binary
*.woff binary
*.woff2 binary
*.otf binary
*.pyc binary
*.pyd binary
*.pyo binary
*.so binary
*.dll binary
*.exe binary
EOF
    
    # Create LICENSE file if it doesn't exist
    if [ ! -f "LICENSE" ]; then
        print_step "Creating LICENSE file..."
        cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Stanley Chisango (Scooter Lacroix)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
    fi
    
    # Create CONTRIBUTING.md
    print_step "Creating CONTRIBUTING.md..."
    cat > CONTRIBUTING.md << 'EOF'
# Contributing to Stan's ML Stack

Thank you for considering contributing to Stan's ML Stack! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -m "Add your meaningful commit message here"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Submit a pull request**

## Coding Standards

- Follow PEP 8 for Python code
- Use shellcheck for shell scripts
- Include comments and documentation
- Add tests for new features

## Pull Request Process

1. Ensure your code follows the coding standards
2. Update the documentation if necessary
3. Include tests for new features
4. Ensure all tests pass
5. Submit the pull request

## Reporting Issues

If you find a bug or have a suggestion for improvement, please create an issue on the GitHub repository. Please include:

- A clear and descriptive title
- A detailed description of the issue or suggestion
- Steps to reproduce the issue (if applicable)
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- System information (OS, Python version, etc.)

## Contact

If you have any questions or need help, feel free to contact the maintainer:

- Email: scooterlacroix@gmail.com
- GitHub: https://github.com/scooter-lacroix
- X: https://x.com/scooter_lacroix
- Patreon: https://patreon.com/ScooterLacroix

Thank you for your contribution!
EOF
    
    print_success "Git files created"
}

# Function to prepare Git repository
prepare_git_repository() {
    print_section "Preparing Git repository"
    
    # Check if Git is installed
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git and try again."
        return 1
    fi
    
    # Check if directory is already a Git repository
    if [ -d ".git" ]; then
        print_warning "Directory is already a Git repository."
        return 0
    fi
    
    # Initialize Git repository
    print_step "Initializing Git repository..."
    git init
    
    # Add files to Git
    print_step "Adding files to Git..."
    git add .
    
    # Create initial commit
    print_step "Creating initial commit..."
    git commit -m "Initial commit of Stan's ML Stack"
    
    print_success "Git repository prepared"
    
    # Print next steps
    print_step "Next steps:"
    echo "1. Create a repository on GitHub or another Git hosting service"
    echo "2. Add the remote repository:"
    echo "   git remote add origin https://github.com/username/repository.git"
    echo "3. Push the code to the remote repository:"
    echo "   git push -u origin main"
}

# Main function
main() {
    print_header "ML Stack Git Preparation"
    
    # Clean up temporary files
    cleanup_temp_files
    
    # Organize directory structure
    organize_directory_structure
    
    # Create Git files
    create_git_files
    
    # Ask if user wants to prepare Git repository
    read -p "Do you want to prepare the Git repository now? (y/n): " prepare_git
    if [ "$prepare_git" = "y" ] || [ "$prepare_git" = "Y" ]; then
        prepare_git_repository
    else
        print_step "Skipping Git repository preparation."
        print_step "To prepare the Git repository later, run:"
        echo "git init"
        echo "git add ."
        echo "git commit -m \"Initial commit of Stan's ML Stack\""
    fi
    
    print_header "ML Stack Git Preparation Complete"
}

# Run main function
main
