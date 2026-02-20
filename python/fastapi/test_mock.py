from unittest.mock import Mock, patch, MagicMock

mock = MagicMock()
print(type(mock))
mock.feature_names = [f'V{i}' for i in range(1, 29)]
mock.detect_drift.return_value = {
    'overall_psi': 0.05,
    'drift_detected': False
}
print("="*50)
print(type(mock))
print(mock)
print(f"Mock values: {mock.feature_names}")
print(f"Mock detect_drift: {mock.detect_drift.return_value}")
print("="*50)
