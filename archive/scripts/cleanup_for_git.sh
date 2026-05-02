#!/bin/bash
#
# Script to clean up the ML Stack directory for Git
# This script identifies files that shouldn't be in the Git repository,
# moves them to a separate directory, and updates .gitignore
#

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RESET='\033[0m'
BOLD='\033[1m'

# Print header
echo -e "${BLUE}${BOLD}=== Cleaning Up ML Stack Directory for Git ===${RESET}\n"

# Create excluded directory
EXCLUDED_DIR="$HOME/Desktop/Stans_MLStack/excluded"
mkdir -p "$EXCLUDED_DIR"
echo -e "${GREEN}✓ Created excluded directory: $EXCLUDED_DIR${RESET}"

# Define patterns for files to exclude
EXCLUDE_PATTERNS=(
    "*.pyc"
    "*.pyo"
    "*.pyd"
    "__pycache__"
    "*.so"
    "*.o"
    "*.a"
    "*.lib"
    "*.dll"
    "*.dylib"
    "*.exe"
    "*.out"
    "*.app"
    "*.log"
    "*.tmp"
    "*.temp"
    "*.swp"
    "*.swo"
    ".DS_Store"
    "Thumbs.db"
    "build/"
    "dist/"
    "*.egg-info/"
    ".eggs/"
    ".pytest_cache/"
    ".coverage"
    "htmlcov/"
    ".tox/"
    ".env"
    ".venv"
    "env/"
    "venv/"
    "ENV/"
    "env.bak/"
    "venv.bak/"
    ".ipynb_checkpoints"
    "*.onnx"
    "*.pt"
    "*.pth"
    "*.bin"
    "*.h5"
    "*.hdf5"
    "*.pb"
    "*.tflite"
    "*.mlmodel"
    "*.csv"
    "*.tsv"
    "*.json"
    "*.jsonl"
    "*.parquet"
    "*.arrow"
    "*.feather"
    "*.pickle"
    "*.pkl"
    "*.npy"
    "*.npz"
)

# Define directories that should be excluded entirely
EXCLUDE_DIRS=(
    "onnxruntime_build"
    "vllm_build"
    "benchmark_results"
    "test_results"
    "logs"
    "data"
    "models"
)

# Function to check if a file matches any exclude pattern
matches_exclude_pattern() {
    local file="$1"
    
    # Check if file is in an excluded directory
    for dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$file" == *"$dir"* ]]; then
            return 0
        fi
    done
    
    # Check if file matches an exclude pattern
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$file" == *$pattern* ]]; then
            return 0
        fi
    done
    
    return 1
}

# Function to check if a file is a build artifact or temporary file
is_build_artifact() {
    local file="$1"
    
    # Check file extension
    if [[ "$file" == *.o || "$file" == *.so || "$file" == *.a || "$file" == *.pyc || 
          "$file" == *.pyo || "$file" == *.pyd || "$file" == *.dll || "$file" == *.dylib || 
          "$file" == *.exe || "$file" == *.out || "$file" == *.app || "$file" == *.log || 
          "$file" == *.tmp || "$file" == *.temp || "$file" == *.swp || "$file" == *.swo ]]; then
        return 0
    fi
    
    # Check directory name
    if [[ "$file" == *"/build/"* || "$file" == *"/dist/"* || "$file" == *"/__pycache__/"* || 
          "$file" == *"/.pytest_cache/"* || "$file" == *"/.tox/"* || "$file" == *"/.eggs/"* || 
          "$file" == *"/.ipynb_checkpoints/"* ]]; then
        return 0
    fi
    
    return 1
}

# Function to check if a file is a large binary file
is_large_binary() {
    local file="$1"
    
    # Check file extension
    if [[ "$file" == *.onnx || "$file" == *.pt || "$file" == *.pth || "$file" == *.bin || 
          "$file" == *.h5 || "$file" == *.hdf5 || "$file" == *.pb || "$file" == *.tflite || 
          "$file" == *.mlmodel ]]; then
        return 0
    fi
    
    # Check if file is larger than 10MB
    local size=$(stat -c %s "$file" 2>/dev/null || echo 0)
    if (( size > 10485760 )); then
        return 0
    fi
    
    return 1
}

# Function to check if a file is a data file
is_data_file() {
    local file="$1"
    
    # Check file extension
    if [[ "$file" == *.csv || "$file" == *.tsv || "$file" == *.json || "$file" == *.jsonl || 
          "$file" == *.parquet || "$file" == *.arrow || "$file" == *.feather || "$file" == *.pickle || 
          "$file" == *.pkl || "$file" == *.npy || "$file" == *.npz || "$file" == *.h5 || 
          "$file" == *.hdf5 ]]; then
        return 0
    fi
    
    return 1
}

# Find all files in the ML Stack directory
echo -e "${BLUE}>> Finding files to exclude...${RESET}"
EXCLUDED_FILES=()
MLSTACK_DIR="$HOME/Desktop/Stans_MLStack"

while IFS= read -r file; do
    # Skip the excluded directory itself
    if [[ "$file" == "$EXCLUDED_DIR"* ]]; then
        continue
    fi
    
    # Check if file should be excluded
    if matches_exclude_pattern "$file" || is_build_artifact "$file" || is_large_binary "$file" || is_data_file "$file"; then
        EXCLUDED_FILES+=("$file")
    fi
done < <(find "$MLSTACK_DIR" -type f -not -path "$EXCLUDED_DIR/*" 2>/dev/null)

# Move excluded files to the excluded directory
if [ ${#EXCLUDED_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}>> Moving ${#EXCLUDED_FILES[@]} excluded files to $EXCLUDED_DIR...${RESET}"
    
    for file in "${EXCLUDED_FILES[@]}"; do
        # Create relative path within excluded directory
        rel_path="${file#$MLSTACK_DIR/}"
        target_dir="$EXCLUDED_DIR/$(dirname "$rel_path")"
        
        # Create target directory if it doesn't exist
        mkdir -p "$target_dir"
        
        # Move file
        mv "$file" "$target_dir/"
        echo -e "  ${YELLOW}→ Moved: $rel_path${RESET}"
    done
    
    echo -e "${GREEN}✓ Moved ${#EXCLUDED_FILES[@]} excluded files to $EXCLUDED_DIR${RESET}"
else
    echo -e "${GREEN}✓ No files to exclude${RESET}"
fi

# Update .gitignore
echo -e "${BLUE}>> Updating .gitignore...${RESET}"
GITIGNORE_FILE="$MLSTACK_DIR/.gitignore"

# Add excluded directory to .gitignore
if ! grep -q "^excluded/" "$GITIGNORE_FILE"; then
    echo "excluded/" >> "$GITIGNORE_FILE"
    echo -e "${GREEN}✓ Added excluded/ to .gitignore${RESET}"
fi

# Add excluded directories to .gitignore
for dir in "${EXCLUDE_DIRS[@]}"; do
    if ! grep -q "^$dir/" "$GITIGNORE_FILE"; then
        echo "$dir/" >> "$GITIGNORE_FILE"
        echo -e "${GREEN}✓ Added $dir/ to .gitignore${RESET}"
    fi
done

# Add excluded patterns to .gitignore
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    if ! grep -q "^$pattern$" "$GITIGNORE_FILE"; then
        echo "$pattern" >> "$GITIGNORE_FILE"
        echo -e "${GREEN}✓ Added $pattern to .gitignore${RESET}"
    fi
done

# Sort and deduplicate .gitignore
sort -u "$GITIGNORE_FILE" -o "$GITIGNORE_FILE.tmp"
mv "$GITIGNORE_FILE.tmp" "$GITIGNORE_FILE"
echo -e "${GREEN}✓ Sorted and deduplicated .gitignore${RESET}"

# Create a list of files that will be included in the Git repository
echo -e "${BLUE}>> Creating list of files to include in Git repository...${RESET}"
INCLUDED_FILES_LIST="$MLSTACK_DIR/git_included_files.txt"

find "$MLSTACK_DIR" -type f -not -path "$EXCLUDED_DIR/*" -not -path "*/\.*" | sort > "$INCLUDED_FILES_LIST"

echo -e "${GREEN}✓ Created list of files to include in Git repository: $INCLUDED_FILES_LIST${RESET}"
echo -e "${YELLOW}>> Review this list to ensure all necessary files are included${RESET}"

# Create a list of directories in the repository
echo -e "${BLUE}>> Creating list of directories in the repository...${RESET}"
DIRS_LIST="$MLSTACK_DIR/git_directories.txt"

find "$MLSTACK_DIR" -type d -not -path "$EXCLUDED_DIR/*" -not -path "*/\.*" | sort > "$DIRS_LIST"

echo -e "${GREEN}✓ Created list of directories in the repository: $DIRS_LIST${RESET}"

# Create a summary of the cleanup
echo -e "${BLUE}>> Creating cleanup summary...${RESET}"
SUMMARY_FILE="$MLSTACK_DIR/git_cleanup_summary.txt"

{
    echo "ML Stack Git Cleanup Summary"
    echo "==========================="
    echo
    echo "Excluded Directory: $EXCLUDED_DIR"
    echo "Number of Excluded Files: ${#EXCLUDED_FILES[@]}"
    echo
    echo "Directory Structure:"
    find "$MLSTACK_DIR" -type d -not -path "$EXCLUDED_DIR/*" -not -path "*/\.*" | sort | sed 's|'"$MLSTACK_DIR"'|.|g' | sed 's|^\.\/|  |g' | sed 's|\/|/|g'
    echo
    echo "Files to be included in Git repository: $(wc -l < "$INCLUDED_FILES_LIST")"
    echo
    echo "Excluded Patterns:"
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        echo "  - $pattern"
    done
    echo
    echo "Excluded Directories:"
    for dir in "${EXCLUDE_DIRS[@]}"; do
        echo "  - $dir/"
    done
} > "$SUMMARY_FILE"

echo -e "${GREEN}✓ Created cleanup summary: $SUMMARY_FILE${RESET}"

echo -e "\n${GREEN}${BOLD}=== ML Stack Directory Cleanup Complete ===${RESET}"
echo -e "Please review the following files:"
echo -e "  - ${YELLOW}$INCLUDED_FILES_LIST${RESET} - List of files to include in Git repository"
echo -e "  - ${YELLOW}$DIRS_LIST${RESET} - List of directories in the repository"
echo -e "  - ${YELLOW}$SUMMARY_FILE${RESET} - Cleanup summary"
echo -e "  - ${YELLOW}$GITIGNORE_FILE${RESET} - Updated .gitignore file"
echo -e "\nExcluded files have been moved to: ${YELLOW}$EXCLUDED_DIR${RESET}"
echo -e "You can restore these files if needed."
