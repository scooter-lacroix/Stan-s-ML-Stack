# Shell Scripting Style Guide

A comprehensive guide for writing clean, maintainable, and portable shell scripts following modern best practices (2025/2026).

## Table of Contents

- [Shebang and File Format](#shebang-and-file-format)
- [Naming Conventions](#naming-conventions)
- [Variables and Parameters](#variables-and-parameters)
- [Quoting and Escaping](#quoting-and-escaping)
- [Conditionals](#conditionals)
- [Loops](#loops)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Command Substitution](#command-substitution)
- [Input/Output](#inputoutput)
- [Best Practices](#best-practices)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Shebang and File Format

### Shebang

```bash
#!/usr/bin/env bash

# Good: Use env for portability
# This finds bash in the user's PATH

# Bad: Hardcoded path
#!/bin/bash

# Good: Specify bash version requirements (if needed)
# Requires bash 4.0 or higher
if [[ ${BASH_VERSION%%.*} -lt 4 ]]; then
  echo "Error: Bash 4.0 or higher required" >&2
  exit 1
fi
```

### File Encoding and Line Endings

```bash
# Good: UTF-8 encoding, Unix line endings (LF)
# Set in editor or use:
# dos2unix script.sh

# Good: Add encoding declaration if using non-ASCII
# -*- coding: utf-8 -*-
```

---

## Naming Conventions

### Variable Names

```bash
# Good: Lowercase with underscores for variables
user_name="John"
file_count=10
is_active=true

# Bad: Mixed case or special characters
userName="John"        # Inconsistent with shell style
file-count=10          # Hyphens can cause issues
2count=10              # Can't start with number

# Good: Uppercase for constants and environment variables
readonly MAX_RETRIES=5
readonly API_BASE_URL="https://api.example.com"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Good: Prefix with underscore for "private" variables
_private_function() {
  local _internal_var="something"
}
```

### Function Names

```bash
# Good: lowercase_with_underscores
create_user() {
  echo "Creating user: $1"
}

check_dependencies() {
  echo "Checking dependencies..."
}

# Good: Verbs for actions
install_package
start_service
stop_service
restart_service
check_status

# Bad: Unclear names
process()        # What does it process?
do_it()          # Too vague
x()              # Meaningless
```

---

## Variables and Parameters

### Variable Declaration

```bash
# Good: Use local for function-local variables
my_function() {
  local user_name="John"
  local file_count=0

  # Process files...
  file_count=$(ls -1 | wc -l)
}

# Good: Use readonly for constants
readonly CONFIG_FILE="/etc/myapp/config.conf"
readonly MAX_CONNECTIONS=100

# Good: Declare arrays explicitly
declare -a files=(
  "file1.txt"
  "file2.txt"
  "file3.txt"
)

# Good: Declare associative arrays (bash 4+)
declare -A user_info=(
  [name]="John Doe"
  [email]="john@example.com"
  [age]=30
)

# Bad: Unintentional globals
bad_function() {
  result=42  # Creates global variable
}

# Good: Explicit locals
good_function() {
  local result=42
  echo "$result"
}
```

### Parameter Expansion

```bash
# Good: Use parameter expansion for defaults
name="${1:-"Unknown"}"           # Default if unset or null
count="${2:-10}"                 # Default to 10
file="${3:-"/tmp/default.txt"}"  # Default path

# Good: Use := for assignment if unset or null
name="${1:="Anonymous"}"  # Sets name if unset/null AND assigns it

# Good: Use :- to default without assignment
echo "User: ${1:-"Unknown"}"  # Does NOT set $1

# Good: Error on unset
name="${1:?"Error: name parameter required"}"

# Good: String length
path="/usr/local/bin"
echo "Length: ${#path}"  # Output: Length: 14

# Good: Substring extraction
filename="document.txt"
echo "${filename:0:7}"   # Output: "documen"
echo "${filename: -4}"   # Output: ".txt" (from end)

# Good: Remove patterns
path="/path/to/file.txt"
echo "${path%.txt}"      # Output: /path/to/file
echo "${path##*/}"       # Output: file.txt
echo "${path%/*}"        # Output: /path/to

# Good: Search and replace
text="Hello, World!"
echo "${text/World/Bash}"        # Output: Hello, Bash!
echo "${text//l/L}"              # Output: HeLLo, WorLd!
```

---

## Quoting and Escaping

### Double Quotes

```bash
# Good: Use double quotes for variable expansion
name="John"
echo "Hello, $name"      # Output: Hello, John

# Good: Use double quotes to prevent word splitting
files="file1.txt file2.txt"
for file in $files; do  # Bad: splits on spaces
  echo "$file"
done

files=("file1.txt" "file2.txt")
for file in "${files[@]}"; do  # Good: proper array handling
  echo "$file"
done

# Good: Double quotes preserve whitespace
message="  Hello, World!  "
echo "$message"  # Output:   Hello, World!
```

### Single Quotes

```bash
# Good: Use single quotes for literal strings
echo 'Hello, $USER'      # Output: Hello, $USER (literal)
echo 'It\'s me'         # Output: It's me

# Good: Single quotes prevent all expansion
echo 'The value is $((5 + 3))'  # Output: The value is $((5 + 3))
```

### Escaping

```bash
# Good: Escape special characters
echo "Price: \$100"      # Output: Price: $100
echo "Path: C:\\Users"   # Output: Path: C:\Users
echo "Quote: \""         # Output: Quote: "

# Good: Use $'...' for ANSI-C quoting
echo $'Hello\tWorld!'    # Output: Hello    World! (with tab)
echo $'Line1\nLine2'     # Output: Line1<newline>Line2
```

---

## Conditionals

### Test Commands

```bash
# Good: Use [[ for string comparisons
if [[ "$name" == "John" ]]; then
  echo "Hello, John"
fi

# Good: Use (( for arithmetic
if (( count > 10 )); then
  echo "Count is greater than 10"
fi

# Good: Check if variable is set
if [[ -z "${name:-}" ]]; then
  echo "Name is not set"
fi

# Good: Pattern matching
if [[ "$filename" == *.txt ]]; then
  echo "Text file"
fi

# Good: Regex matching (bash 3+)
if [[ "$email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
  echo "Valid email"
fi
```

### File Tests

```bash
# Good: File existence checks
if [[ -f "$file" ]]; then
  echo "File exists"
fi

if [[ -d "$dir" ]]; then
  echo "Directory exists"
fi

# Good: Multiple file tests
if [[ -f "$file" && -r "$file" && -w "$file" ]]; then
  echo "File exists and is readable/writable"
fi

# Common file operators
-f "$file"    # Regular file exists
-d "$dir"     # Directory exists
-e "$path"    # File/directory exists
-r "$file"    # File is readable
-w "$file"    # File is writable
-x "$file"    # File is executable
-s "$file"    # File is not empty
-L "$link"    # Symbolic link
```

### String Tests

```bash
# Good: String comparisons
if [[ "$str1" == "$str2" ]]; then
  echo "Strings are equal"
fi

if [[ "$str1" != "$str2" ]]; then
  echo "Strings are different"
fi

if [[ -z "$string" ]]; then
  echo "String is empty"
fi

if [[ -n "$string" ]]; then
  echo "String is not empty"
fi

# Good: Lexicographical comparison
if [[ "$str1" < "$str2" ]]; then
  echo "str1 comes before str2"
fi
```

---

## Loops

### For Loops

```bash
# Good: Iterate over explicit list
for file in file1.txt file2.txt file3.txt; do
  echo "Processing: $file"
done

# Good: Iterate over array
files=("file1.txt" "file2.txt" "file3.txt")
for file in "${files[@]}"; do
  echo "Processing: $file"
done

# Good: Iterate over command output
for file in *.txt; do
  echo "Found: $file"
done

# Good: C-style for loop
for ((i=0; i<10; i++)); do
  echo "Iteration: $i"
done

# Good: Iterate over lines
while IFS= read -r line; do
  echo "Line: $line"
done < "input.txt"
```

### While Loops

```bash
# Good: While with condition
count=0
while (( count < 10 )); do
  echo "Count: $count"
  ((count++))
done

# Good: Read file line by line
while IFS= read -r line; do
  echo "Processing: $line"
done < "input.txt"

# Good: Process command output
while IFS= read -r line; do
  echo "Line: $line"
done < <(find . -type f -name "*.txt")

# Good: Infinite loop with break
while true; do
  read -rp "Enter a name (or 'quit'): " name
  [[ "$name" == "quit" ]] && break
  echo "Hello, $name"
done
```

### Loop Control

```bash
# Good: Continue to next iteration
for i in {1..10}; do
  if (( i % 2 == 0 )); then
    continue
  fi
  echo "Odd number: $i"
done

# Good: Break from loop
for file in *.txt; do
  if [[ "$file" == "stop.txt" ]]; then
    break
  fi
  echo "Processing: $file"
done

# Good: Break from nested loops
for dir in */; do
  for file in "$dir"*.txt; do
    if [[ "$file" == *"skip"* ]]; then
      break 2  # Break out of both loops
    fi
    echo "Processing: $file"
  done
done
```

---

## Functions

### Function Definition

```bash
# Good: Modern function syntax
my_function() {
  local param1="$1"
  local param2="${2:-"default"}"
  local result

  # Function body
  result=$((param1 + param2))

  echo "$result"
}

# Good: Function with multiple parameters
process_data() {
  local input_file="$1"
  local output_file="${2:-"output.txt"}"
  local verbose="${3:-false}"

  if [[ ! -f "$input_file" ]]; then
    return 1
  fi

  # Process data...
  echo "Processed data written to $output_file"
  return 0
}

# Good: Function returning array
get_files() {
  local pattern="$1"
  local -a files

  files=($(find . -name "$pattern"))
  echo "${files[@]}"
}

# Usage
files=($(get_files "*.txt"))
echo "Found ${#files[@]} files"
```

### Function Return Values

```bash
# Good: Return 0 for success, non-zero for failure
check_file() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
    return 1
  fi

  if [[ ! -r "$file" ]]; then
    return 2
  fi

  return 0
}

# Usage
if check_file "myfile.txt"; then
  echo "File is valid"
else
  echo "File check failed with code: $?"
fi

# Good: Echo output for data retrieval
get_username() {
  local user_id="$1"

  # In real script, might query database
  case "$user_id" in
    1) echo "john" ;;
    2) echo "jane" ;;
    *) echo "unknown" ;;
  esac
}

# Usage
username=$(get_username 1)
echo "Username: $username"
```

---

## Error Handling

### Set Options

```bash
# Good: Use set -e for immediate exit on error
set -e  # Exit on error
set -u  # Exit on unset variable
set -o pipefail  # Exit on pipe failure

# Good: Combined
set -euo pipefail

# Good: Trap errors
trap 'echo "Error on line $LINENO"; exit 1' ERR

# Good: Trap exit
trap 'cleanup; echo "Script exiting";' EXIT

# Good: Cleanup function
cleanup() {
  local exit_code=$?

  echo "Performing cleanup..."

  # Remove temp files, etc.
  [[ -f "${tmp_file:-}" ]] && rm -f "$tmp_file"

  exit $exit_code
}

trap cleanup EXIT
```

### Error Checking

```bash
# Good: Check command success
if ! command -v git &> /dev/null; then
  echo "Error: git is not installed" >&2
  exit 1
fi

# Good: Check file operations
if ! cp "$source" "$destination"; then
  echo "Error: Failed to copy file" >&2
  exit 1
fi

# Good: Check with and/or
cd /path/to/dir || { echo "Cannot cd to directory"; exit 1; }
mkdir -p "$dir" && echo "Directory created"

# Good: Validate inputs
validate_email() {
  local email="$1"

  if [[ ! "$email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
    echo "Error: Invalid email address: $email" >&2
    return 1
  fi

  return 0
}
```

---

## Command Substitution

### Backticks vs $()

```bash
# Good: Use $() for command substitution
files=$(ls -1)
echo "Files: $files"

# Bad: Use backticks (deprecated)
files=`ls -1`

# Good: Nesting is easier with $()
count=$(wc -l < $(find . -name "*.txt"))
```

### Here Documents

```bash
# Good: Here document for multi-line strings
cat <<EOF
This is a multi-line
string that preserves
formatting.
EOF

# Good: Here document with variable expansion
name="John"
cat <<EOF
Hello, $name!
Today is $(date +%Y-%m-%d)
EOF

# Good: Here document without expansion (single quotes)
cat <<'EOF'
This text is literal.
$USER and $(date) won't be expanded.
EOF

# Good: Here string for single input
grep "pattern" <<< "This is the input string"

# Good: Redirect heredoc to file
cat > output.txt <<EOF
Line 1
Line 2
Line 3
EOF
```

---

## Input/Output

### Reading Input

```bash
# Good: Read user input
read -rp "Enter your name: " name
echo "Hello, $name"

# Good: Read password (no echo)
read -rsp "Enter password: " password
echo

# Good: Read into array
read -rp "Enter names (space separated): " -a names
echo "Names: ${names[@]}"

# Good: Read with timeout
read -rp "Press any key (5 seconds)... " -t 5 -n 1
echo

# Good: Read file line by line
while IFS= read -r line; do
  echo "Line: $line"
done < "input.txt"
```

### Output Formatting

```bash
# Good: Use printf for formatting
printf "Name: %-20s Age: %3d\n" "John" 30
printf "Name: %-20s Age: %3d\n" "Jane" 25

# Good: Format numbers
printf "Pi: %.2f\n" 3.14159
printf "Scientific: %e\n" 1000000

# Good: Column output
printf "%-20s %-10s %s\n" "Name" "Age" "Email"
printf "%-20s %-10d %s\n" "John Doe" 30 "john@example.com"

# Good: Use here documents for tables
cat <<EOF
Name            Age    Email
John Doe        30     john@example.com
Jane Smith      25     jane@example.com
EOF
```

---

## Best Practices

### Script Header

```bash
#!/usr/bin/env bash

# ==============================================================================
# Script Name: deploy.sh
# Description: Automated deployment script for web applications
# Author: Your Name
# Date: 2025-01-05
# Version: 1.0.0
# Usage: ./deploy.sh [environment] [branch]
# ==============================================================================

set -euo pipefail

# Script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration files
readonly CONFIG_FILE="${SCRIPT_DIR}/config/deploy.conf"
readonly LOG_FILE="${SCRIPT_DIR}/logs/deploy.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
  local level="$1"
  shift
  local message="[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*"
  echo "$message" | tee -a "$LOG_FILE"
}

# Main function
main() {
  local environment="${1:-"production"}"
  local branch="${2:-"main"}"

  log "INFO" "Starting deployment"
  log "INFO" "Environment: $environment"
  log "INFO" "Branch: $branch"

  # Deployment logic here...

  log "INFO" "Deployment complete"
}

# Run main function
main "$@"
```

### Portability

```bash
# Good: Check required commands
check_requirements() {
  local required_commands=("git" "docker" "curl")
  local missing_commands=()

  for cmd in "${required_commands[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
      missing_commands+=("$cmd")
    fi
  done

  if (( ${#missing_commands[@]} > 0 )); then
    echo "Error: Missing required commands: ${missing_commands[*]}" >&2
    return 1
  fi

  return 0
}

# Good: Check OS compatibility
if [[ "$OSTYPE" != "linux-gnu"* ]] && [[ "$OSTYPE" != "darwin"* ]]; then
  echo "Error: Unsupported OS: $OSTYPE" >&2
  exit 1
fi
```

---

## Anti-Patterns to Avoid

### Don't Parse ls

```bash
# Bad: Parsing ls output
for file in $(ls); do  # Breaks on filenames with spaces
  echo "$file"
done

# Good: Use globbing
for file in *; do
  echo "$file"
done

# Good: Use find
find . -type f -print0 | while IFS= read -r -d '' file; do
  echo "$file"
done
```

### Don't Use Echo for Variables

```bash
# Bad: Using echo to assign variables
files=$(echo *.txt)  # Fails with no matches

# Good: Direct assignment or arrays
files=(*.txt)       # Array, handles empty case
```

---

## Additional Resources

- [Bash Guide for Beginners](https://tldp.org/LDP/Bash-Beginners-Guide/html/)
- [Advanced Bash-Scripting Guide](https://tldp.org/LDP/abs/html/)
- [ShellCheck](https://www.shellcheck.net/)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
