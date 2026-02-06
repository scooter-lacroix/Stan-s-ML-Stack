import sys
import os
import re

path = 'core/flash_attention/CMakeLists.txt'
if not os.path.exists(path):
    print(f"Error: {path} not found")
    sys.exit(1)

with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    if 'ROCM_VERSION_MATCH ${ROCM_VERSION_DEV_RAW}' in line and 'REGEX MATCH' in line:
        # We want to replace the strict regex with one that matches 7.2.0 correctly
        new_line = re.sub(r'"\^.*\$\"', r'"^([0-9]+)\\.([0-9]+)\\.([0-9]+).*"', line)
        new_lines.append(new_line)
        if new_line != line:
            found = True
    else:
        new_lines.append(line)

if not found:
    print("Warning: Could not find/patch ROCM version regex in CMakeLists.txt")

with open(path, 'w') as f:
    f.writelines(new_lines)
