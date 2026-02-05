import re

path = 'rusty-stack/src/component_status.rs'
with open(path, 'r') as f:
    content = f.read()

# Correct snippets with proper semicolon/if handling
onnx_clean = "import onnxruntime as ort; import pathlib; import os; import sys; print('Version:', ort.__version__); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob('libonnxruntime_providers_rocm.so')); os.environ.update({'ORT_ROCM_EP_PROVIDER_PATH': str(libs[0])} if libs else {}); providers=ort.get_available_providers(); print('Providers:', providers); sys.exit(0 if 'ROCMExecutionProvider' in providers else 1)"
bnb_clean = "import bitsandbytes as bnb; import pathlib; import sys; print('Version:', getattr(bnb, '__version__', 'unknown')); path=pathlib.Path(bnb.__file__).parent; libs=list(path.glob('libbitsandbytes_rocm*.so')); print('ROCm Libs:', libs); sys.exit(0 if libs else 1)"

# Match any occurrence of the complex strings
content = re.sub(r'import onnxruntime as ort;.*?in providers else 1\)', onnx_clean, content)
content = re.sub(r'import bitsandbytes as bnb;.*?libs else 1\)', bnb_clean, content)

with open(path, 'w') as f:
    f.write(content)
