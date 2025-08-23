"""
Numpy compatibility utilities for handling different numpy versions.
"""
import sys
import numpy as np
import types


def setup_numpy_compatibility():
    """
    Set up numpy compatibility for different versions.
    
    This handles the numpy._core.numeric compatibility issue that occurs
    when loading pickle files created with different numpy versions.
    """
    try:
        # First, ensure numpy._core exists
        if not hasattr(np, '_core'):
            np._core = types.ModuleType('_core')
            sys.modules['numpy._core'] = np._core
        
        # Create numpy._core.numeric module if it doesn't exist
        if not hasattr(np._core, 'numeric'):
            # Try to use the actual numeric module if it exists
            if hasattr(np, 'core') and hasattr(np.core, 'numeric'):
                np._core.numeric = np.core.numeric
            else:
                # Create a dummy numeric module that points to numpy itself
                np._core.numeric = types.ModuleType('numeric')
                # Copy essential attributes from numpy to the numeric module
                for attr in ['array', 'ndarray', 'dtype', 'float64', 'int64']:
                    if hasattr(np, attr):
                        setattr(np._core.numeric, attr, getattr(np, attr))
            
            sys.modules['numpy._core.numeric'] = np._core.numeric
        
        # Also ensure numpy.core.numeric exists for backward compatibility
        if hasattr(np, 'core') and not hasattr(np.core, 'numeric'):
            np.core.numeric = np._core.numeric
            
    except (AttributeError, ImportError) as e:
        # If compatibility setup fails, create minimal structure
        try:
            if 'numpy._core' not in sys.modules:
                _core_module = types.ModuleType('_core')
                sys.modules['numpy._core'] = _core_module
            
            if 'numpy._core.numeric' not in sys.modules:
                numeric_module = types.ModuleType('numeric')
                # Add essential numpy attributes to the numeric module
                for attr in ['array', 'ndarray', 'dtype', 'float64', 'int64']:
                    if hasattr(np, attr):
                        setattr(numeric_module, attr, getattr(np, attr))
                sys.modules['numpy._core.numeric'] = numeric_module
        except Exception:
            # Last resort: just continue without compatibility
            pass


# Call setup on import
setup_numpy_compatibility()