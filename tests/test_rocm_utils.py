import unittest
import subprocess
import json
from unittest.mock import patch, MagicMock

# Adjust the import path based on your project structure
# Assuming 'core' is a top-level directory accessible in PYTHONPATH
# or the tests are run from a location where 'core' can be found.
from core.rocm.rocm_utils import get_gpu_info, logger

# Disable logging for tests to keep output clean, or configure as needed
logger.setLevel(logging.CRITICAL)

class TestGetGpuInfo(unittest.TestCase):

    @patch('core.rocm.rocm_utils.subprocess.run')
    def test_get_gpu_info_rdna3_success(self, mock_subprocess_run):
        # This is what the mocked subprocess.run should return as stdout
        mock_json_data = [
            {
                "Card name": "Radeon RX 7900 XTX",
                "Card vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                "GPU memory": {"total": "24560 MB"},
                "Temperature": {"edge": "50.0c"},
                "Clocks": {"sclk": "2500MHz"},
                "Fan speed": "30 %",
                "Power": {"average": "150.0 W"}
            },
            {
                "Card name": "Radeon RX 6800 XT",
                "Card vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                "GPU memory": {"total": "16384 MB"},
                "Temperature": {"edge": "45.0c"},
                "Clocks": {"sclk": "2200MHz"},
                "Fan speed": "25 %",
                "Power": {"average": "120.0 W"}
            }
        ]
        mock_json_output = json.dumps(mock_json_data)

        # Configure the mock for subprocess.run
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = mock_json_output
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Call the function
        gpu_info_result = get_gpu_info()

        # Assertions
        self.assertIsNotNone(gpu_info_result, "get_gpu_info should return data, not None")
        self.assertEqual(len(gpu_info_result), 2, "Should parse two GPU entries")

        # Check RDNA3 GPU details (first GPU in mock data)
        rdna3_gpu = gpu_info_result[0]
        self.assertEqual(rdna3_gpu.get('Card name'), "Radeon RX 7900 XTX")
        self.assertEqual(rdna3_gpu.get('Card vendor'), "Advanced Micro Devices, Inc. [AMD/ATI]")
        
        # Check nested information parsing (as per the structure of get_gpu_info)
        # The get_gpu_info function uses .get('key', 'Unknown') for top-level and 
        # .get('ParentKey', {}).get('ChildKey', 'Unknown') for nested.
        # The mocked JSON data is a list of dicts, so direct access is fine here
        # as gpu_info_result is the direct list of dicts.
        self.assertEqual(rdna3_gpu.get('GPU memory', {}).get('total'), "24560 MB")
        self.assertEqual(rdna3_gpu.get('Temperature', {}).get('edge'), "50.0c")
        self.assertEqual(rdna3_gpu.get('Clocks', {}).get('sclk'), "2500MHz")
        self.assertEqual(rdna3_gpu.get('Fan speed'), "30 %") # Fan speed is not nested in the mock
        self.assertEqual(rdna3_gpu.get('Power', {}).get('average'), "150.0 W")


        # Check that rocm-smi was called correctly
        mock_subprocess_run.assert_called_once_with(
            ["rocm-smi", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

    @patch('core.rocm.rocm_utils.subprocess.run')
    def test_get_gpu_info_rocm_smi_failure(self, mock_subprocess_run):
        # Configure the mock for subprocess.run to simulate a failure
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "rocm-smi command failed"
        mock_subprocess_run.return_value = mock_process

        # Call the function
        gpu_info_result = get_gpu_info()

        # Assertions
        self.assertIsNone(gpu_info_result, "get_gpu_info should return None on rocm-smi failure")
        
        # Check that rocm-smi was called
        mock_subprocess_run.assert_called_once_with(
            ["rocm-smi", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

    @patch('core.rocm.rocm_utils.subprocess.run')
    def test_get_gpu_info_json_decode_error(self, mock_subprocess_run):
        # Configure the mock for subprocess.run to return invalid JSON
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "this is not valid json"
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process

        # Call the function
        gpu_info_result = get_gpu_info()

        # Assertions
        self.assertIsNone(gpu_info_result, "get_gpu_info should return None on JSON decode error")

if __name__ == '__main__':
    unittest.main()
