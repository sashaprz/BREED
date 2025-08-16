"""
NumPy compatibility module for handling pickle files created with different NumPy versions.
This module creates the necessary numpy._core.numeric module structure for backward compatibility.
"""

import sys
import numpy as np
import types

def setup_numpy_compatibility():
    """Set up numpy compatibility for loading pickle files created with numpy 2.x"""
    
    # Always set up to ensure we have all required modules
    # Remove existing modules first to avoid conflicts
    modules_to_remove = ['numpy._core', 'numpy._core.numeric', 'numpy._core.multiarray']
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    try:
        # Create the _core module
        _core_module = types.ModuleType('numpy._core')
        sys.modules['numpy._core'] = _core_module
        
        # Create the numeric submodule
        numeric_module = types.ModuleType('numpy._core.numeric')
        
        # Copy all attributes from numpy to the numeric module (including private ones for pickle compatibility)
        for attr_name in dir(np):
            try:
                attr = getattr(np, attr_name)
                setattr(numeric_module, attr_name, attr)
            except (AttributeError, ImportError):
                # Skip attributes that can't be copied
                pass
        
        # Add some specific attributes that might be needed from numpy.core
        if hasattr(np, 'core'):
            for attr_name in dir(np.core):
                try:
                    attr = getattr(np.core, attr_name)
                    setattr(numeric_module, attr_name, attr)
                except (AttributeError, ImportError):
                    pass
        
        # Add essential pickle functions that might be missing
        if not hasattr(numeric_module, '_frombuffer'):
            # Try to get _frombuffer from numpy.core.numeric
            if hasattr(np, 'core') and hasattr(np.core, 'numeric') and hasattr(np.core.numeric, '_frombuffer'):
                numeric_module._frombuffer = np.core.numeric._frombuffer
            elif hasattr(np, 'frombuffer'):
                numeric_module._frombuffer = np.frombuffer
            else:
                # Create a basic _frombuffer function
                def _frombuffer(buffer, dtype=float, count=-1, offset=0):
                    return np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
                numeric_module._frombuffer = _frombuffer
        
        # Create the multiarray submodule (needed for pickle compatibility)
        multiarray_module = types.ModuleType('numpy._core.multiarray')
        
        # Copy multiarray-related attributes from numpy
        multiarray_attrs = ['ndarray', 'dtype', 'array', 'asarray', 'zeros', 'ones', 'empty']
        for attr_name in multiarray_attrs:
            if hasattr(np, attr_name):
                try:
                    attr = getattr(np, attr_name)
                    setattr(multiarray_module, attr_name, attr)
                except (AttributeError, ImportError):
                    pass
        
        # Also copy from numpy.core.multiarray if it exists
        if hasattr(np, 'core') and hasattr(np.core, 'multiarray'):
            for attr_name in dir(np.core.multiarray):
                try:
                    attr = getattr(np.core.multiarray, attr_name)
                    setattr(multiarray_module, attr_name, attr)
                except (AttributeError, ImportError):
                    pass
        
        # Add essential pickle-related functions that might be missing
        if not hasattr(multiarray_module, '_reconstruct'):
            # Try to get _reconstruct from numpy.core.multiarray
            if hasattr(np, 'core') and hasattr(np.core, 'multiarray') and hasattr(np.core.multiarray, '_reconstruct'):
                multiarray_module._reconstruct = np.core.multiarray._reconstruct
            else:
                # Create a basic _reconstruct function for ndarray compatibility
                def _reconstruct(subtype, shape, dtype):
                    return np.ndarray.__new__(subtype, shape, dtype)
                multiarray_module._reconstruct = _reconstruct
        
        # Set up the module hierarchy
        _core_module.numeric = numeric_module
        _core_module.multiarray = multiarray_module
        sys.modules['numpy._core.numeric'] = numeric_module
        sys.modules['numpy._core.multiarray'] = multiarray_module
        
        # Also add to numpy namespace if not present
        if not hasattr(np, '_core'):
            np._core = _core_module
            
        print("NumPy compatibility setup completed successfully")
        
    except Exception as e:
        print(f"Warning: Could not set up NumPy compatibility: {e}")

# Set up compatibility when this module is imported
setup_numpy_compatibility()