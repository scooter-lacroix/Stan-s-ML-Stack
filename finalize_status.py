import re

path = 'rusty-stack/src/component_status.rs'
with open(path, 'r') as f:
    content = f.read()

# Unified one-liner for ONNX
onnx_snippet = 'import onnxruntime as ort; import pathlib; import os; import sys; print(\'Version:\', ort.__version__); base=pathlib.Path(ort.__file__).parent; libs=list(base.rglob(\'libonnxruntime_providers_rocm.so\')); [os.environ.update({\'ORT_ROCM_EP_PROVIDER_PATH\': str(l)}) for l in libs[:1]]; providers=ort.get_available_providers(); print(\'Providers:\', providers); sys.exit(0 if \'ROCMExecutionProvider\' in providers else 1)'

# Unified one-liner for BNB
bnb_snippet = 'import bitsandbytes as bnb; import pathlib; import sys; print(\'Version:\', getattr(bnb, \'__version__\', \'unknown\')); path=pathlib.Path(bnb.__file__).parent; libs=list(path.glob(\'libbitsandbytes_rocm*.so\')); print(\'ROCm Libs:\', libs); sys.exit(0 if libs else 1)'

# Generic replacement for any variation of the previous long strings
content = re.sub(r'import onnxruntime as ort;.*?in providers else 1\)', onnx_snippet, content, flags=re.DOTALL)
content = re.sub(r'import bitsandbytes as bnb;.*?libs else 1\)', bnb_snippet, content, flags=re.DOTALL)

with open(path, 'w') as f:
    f.write(content)
