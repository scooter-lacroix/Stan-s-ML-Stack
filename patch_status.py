import re

path = 'rusty-stack/src/component_status.rs'
with open(path, 'r') as f:
    content = f.read()

# Correct snippets
onnx_snippet = 'import onnxruntime as ort; import pathlib; import os; import sys; print(f\\"Version: {ort.__version__}\\"); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob(\\"libonnxruntime_providers_rocm.so\\")); os.environ.update({\\"ORT_ROCM_EP_PROVIDER_PATH\\": str(libs[0])} if libs else {}); providers=ort.get_available_providers(); print(f\\"Providers: {providers}\\"); sys.exit(0 if \\"ROCMExecutionProvider\\" in providers else 1)'
bnb_snippet = 'import bitsandbytes as bnb; import pathlib; import sys; print(f\\"Version: {getattr(bnb, \'__version__\', \'unknown\')}\\"); path=pathlib.Path(bnb.__file__).parent; libs=list(path.glob(\'libbitsandbytes_rocm*.so\')); print(f\\"ROCm Libs: {libs}\\"); sys.exit(0 if libs else 1)'

# Replace ONNX snippets
content = re.sub(r'import onnxruntime as ort;.*in providers else 1\)', onnx_snippet, content)
# Replace BNB snippets
content = re.sub(r'import bitsandbytes as bnb;.*libs else 1\)', bnb_snippet, content)

with open(path, 'w') as f:
    f.write(content)
