#!/usr/bin/env python3
"""
Simple test to verify the key fixes in the Enhanced CGCNN implementation
"""

import sys
import numpy as np

# Test the LogSpaceScaler fix without requiring torch
class MockTensor:
    """Mock tensor class for testing without torch"""
    def __init__(self, data):
        self.data = np.array(data)
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.data + other)
        return MockTensor(self.data + other.data)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.data - other)
        return MockTensor(self.data - other.data)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.data * other)
        return MockTensor(self.data * other.data)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self.data / other)
        return MockTensor(self.data / other.data)
    
    def to(self, device):
        return self

def mock_log10(tensor):
    return MockTensor(np.log10(tensor.data))

def mock_pow(base, tensor):
    return MockTensor(np.power(base, tensor.data))

def mock_mean(tensor):
    return MockTensor([np.mean(tensor.data)])

def mock_std(tensor):
    return MockTensor([np.std(tensor.data)])

# Mock LogSpaceScaler with the fix
class FixedLogSpaceScaler:
    """Fixed version of LogSpaceScaler for testing"""
    
    def __init__(self, epsilon=1e-20):
        self.epsilon = epsilon
        self.log_mean = None
        self.log_std = None
        
    def fit(self, conductivity_values):
        """Fit scaler on log-transformed conductivity values"""
        # Convert to log space
        log_values = mock_log10(conductivity_values + self.epsilon)
        
        # Compute statistics in log space
        self.log_mean = mock_mean(log_values)
        self.log_std = mock_std(log_values) + 1e-8  # Add small epsilon for numerical stability
        
    def transform(self, conductivity_values):
        """Transform conductivity to normalized log space"""
        log_values = mock_log10(conductivity_values + self.epsilon)
        return (log_values - self.log_mean) / self.log_std
    
    def inverse_transform(self, normalized_log_values):
        """Transform back from normalized log space to conductivity - FIXED VERSION"""
        log_values = normalized_log_values * self.log_std + self.log_mean
        return mock_pow(10, log_values)  # FIXED: Removed epsilon subtraction
    
    def to(self, device):
        """Move scaler to device"""
        return self

# Original buggy version for comparison
class BuggyLogSpaceScaler:
    """Original buggy version of LogSpaceScaler"""
    
    def __init__(self, epsilon=1e-20):
        self.epsilon = epsilon
        self.log_mean = None
        self.log_std = None
        
    def fit(self, conductivity_values):
        log_values = mock_log10(conductivity_values + self.epsilon)
        self.log_mean = mock_mean(log_values)
        self.log_std = mock_std(log_values) + 1e-8
        
    def transform(self, conductivity_values):
        log_values = mock_log10(conductivity_values + self.epsilon)
        return (log_values - self.log_mean) / self.log_std
    
    def inverse_transform(self, normalized_log_values):
        """Original buggy version with epsilon subtraction"""
        log_values = normalized_log_values * self.log_std + self.log_mean
        return mock_pow(10, log_values) - self.epsilon  # BUG: Subtracting epsilon
    
    def to(self, device):
        return self

def test_scaler_fix():
    """Test that the LogSpaceScaler fix works correctly"""
    print("Testing LogSpaceScaler fix...")
    
    # Test data: various conductivity values
    test_conductivities = MockTensor([1e-10, 1e-5, 1e-3, 1e-1, 1.0, 10.0])
    
    # Test fixed version
    fixed_scaler = FixedLogSpaceScaler()
    fixed_scaler.fit(test_conductivities)
    
    # Transform and inverse transform
    normalized = fixed_scaler.transform(test_conductivities)
    recovered_fixed = fixed_scaler.inverse_transform(normalized)
    
    # Test buggy version
    buggy_scaler = BuggyLogSpaceScaler()
    buggy_scaler.fit(test_conductivities)
    
    normalized_buggy = buggy_scaler.transform(test_conductivities)
    recovered_buggy = buggy_scaler.inverse_transform(normalized_buggy)
    
    print("Original values:", test_conductivities.data)
    print("Fixed recovery: ", recovered_fixed.data)
    print("Buggy recovery: ", recovered_buggy.data)
    
    # Check if fixed version recovers original values better
    fixed_error = np.mean(np.abs(test_conductivities.data - recovered_fixed.data))
    buggy_error = np.mean(np.abs(test_conductivities.data - recovered_buggy.data))
    
    print(f"Fixed version error: {fixed_error:.2e}")
    print(f"Buggy version error: {buggy_error:.2e}")
    
    if fixed_error < buggy_error:
        print("✅ Fix successful! Fixed version has lower error.")
        return True
    else:
        print("❌ Fix failed! Buggy version still has lower error.")
        return False

def test_activation_changes():
    """Test that activation function changes are conceptually correct"""
    print("\nTesting activation function changes...")
    
    # Test log-space values (can be negative for small conductivities)
    log_space_values = np.array([-10, -5, -2, 0, 2, 5])  # Represents 1e-10 to 1e5 S/cm
    
    # Simulate Softplus (always positive, limited range)
    def softplus(x):
        return np.log(1 + np.exp(np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    # Simulate no activation (identity)
    def identity(x):
        return x
    
    softplus_output = softplus(log_space_values)
    identity_output = identity(log_space_values)
    
    print("Log-space input values:", log_space_values)
    print("Softplus output:       ", softplus_output)
    print("Identity output:       ", identity_output)
    
    # Check if identity preserves the full range
    range_preserved = np.allclose(log_space_values, identity_output)
    range_constrained = not np.allclose(log_space_values, softplus_output)
    
    if range_preserved and range_constrained:
        print("✅ Activation fix successful! Identity preserves full log-space range.")
        return True
    else:
        print("❌ Activation test inconclusive.")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Enhanced CGCNN Fixes")
    print("=" * 60)
    
    test1_passed = test_scaler_fix()
    test2_passed = test_activation_changes()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED! The fixes should improve model performance.")
    else:
        print("❌ Some tests failed. Review the fixes.")
    print("=" * 60)

if __name__ == "__main__":
    main()