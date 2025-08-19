# CDVAE Data Caching Guide

This guide explains how to use the caching system implemented for CDVAE to dramatically speed up future training runs.

## Overview

The CDVAE caching system stores preprocessed crystal graph data to disk, eliminating the need to recompute expensive graph representations from CIF files on subsequent runs. This can reduce data loading time from ~5-6 minutes to just a few seconds.

## How It Works

### Automatic Caching
- **Cache Location**: `./data_cache/` directory (created automatically)
- **Cache Keys**: MD5 hashes based on:
  - Data file path, size, and modification time
  - Preprocessing parameters (niggli, primitive, graph_method, properties)
- **Cache Format**: Pickle files with `.pkl` extension
- **Cache Validation**: Automatic cache invalidation when source data changes

### Cache Infrastructure Files
- [`cdvae/common/caching.py`](cdvae/common/caching.py): Core caching utilities
- [`cdvae/common/numpy_compat.py`](cdvae/common/numpy_compat.py): Numpy compatibility for pickle files
- [`cdvae/pl_data/dataset.py`](cdvae/pl_data/dataset.py): Dataset with caching integration

## Usage

### Enabling Caching (Default)
Caching is enabled by default. The first run will be slow as it processes and caches data:

```bash
# First run - processes and caches data (~5-6 minutes)
python cdvae/run.py data.root_path=/path/to/data/mp_20 train.pl_trainer.accelerator=gpu

# Subsequent runs - uses cached data (~10-30 seconds)
python cdvae/run.py data.root_path=/path/to/data/mp_20 train.pl_trainer.accelerator=gpu
```

### Disabling Caching
To disable caching (not recommended for repeated runs):

```bash
python cdvae/run.py data.root_path=/path/to/data/mp_20 train.pl_trainer.accelerator=gpu data.datamodule.datasets.train.use_cache=false
```

### Cache Management

#### Check Cache Status
```python
from cdvae.common.caching import get_data_cache

cache = get_data_cache("./data_cache")
info = cache.info()
print(f"Cache contains {info['num_files']} files")
print(f"Total cache size: {info['total_size_mb']:.1f} MB")
```

#### Clear Cache
```python
from cdvae.common.caching import get_data_cache

cache = get_data_cache("./data_cache")
cache.clear()
print("Cache cleared")
```

Or manually delete the cache directory:
```bash
rm -rf ./data_cache
```

## Performance Benefits

### Before Caching
- **Data Loading Time**: ~5-6 minutes for MP-20 dataset (27,178 structures)
- **Processing Speed**: ~80-100 structures/second
- **Bottleneck**: Crystal graph generation using CrystalNN

### After Caching
- **Data Loading Time**: ~10-30 seconds (99% reduction)
- **Processing Speed**: Limited only by disk I/O and pickle deserialization
- **Benefit**: Immediate training start for subsequent runs

### Example Timeline
```
First Run:
├── Data preprocessing: 5-6 minutes
├── Cache storage: 10-20 seconds  
└── Training: Continues normally

Subsequent Runs:
├── Cache loading: 10-30 seconds
└── Training: Starts immediately
```

## Cache Storage Details

### Cache Directory Structure
```
data_cache/
├── a1b2c3d4e5f6...hash1.pkl  # Train dataset cache
├── f6e5d4c3b2a1...hash2.pkl  # Validation dataset cache
└── 9z8y7x6w5v4u...hash3.pkl  # Test dataset cache
```

### Cache Key Generation
Cache keys are generated from:
```python
# Example cache key components
file_info = f"{data_path}_{file_size}_{modification_time}"
params = f"niggli={niggli}_primitive={primitive}_graph_method={graph_method}_prop={property}"
cache_key = md5(f"{file_info}_{params}").hexdigest()
```

### Cache Invalidation
Cache is automatically invalidated when:
- Source data file is modified (different size or timestamp)
- Preprocessing parameters change
- Cache file is corrupted or unreadable

## Troubleshooting

### Cache Miss Issues
If cache isn't being used when expected:
1. Check if data file was modified
2. Verify preprocessing parameters are identical
3. Check cache directory permissions
4. Look for cache corruption messages in logs

### Storage Space
- Each dataset cache file: ~50-200 MB (depends on dataset size)
- Total cache for MP-20: ~150-600 MB
- Monitor disk space if caching multiple large datasets

### Memory Usage
- Cache loading requires sufficient RAM to hold preprocessed data
- MP-20 dataset: ~2-4 GB RAM during loading
- Consider system memory when caching very large datasets

## Advanced Configuration

### Custom Cache Directory
```python
# In dataset configuration
cache = get_data_cache("/custom/cache/path")
```

### Cache Debugging
Enable verbose cache logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Cache Management
```python
from cdvae.common.caching import DataCache

# Create cache instance
cache = DataCache("./my_cache")

# Store data
cache.set("my_key", my_data)

# Retrieve data
data = cache.get("my_key")

# Get cache info
info = cache.info()
```

## Best Practices

1. **Keep Cache Warm**: Don't delete cache between training runs
2. **Monitor Disk Space**: Cache can grow large with multiple datasets
3. **Backup Important Caches**: For expensive preprocessing operations
4. **Use Consistent Paths**: Absolute paths prevent cache misses
5. **Version Control**: Document preprocessing parameters for reproducibility

## Integration with Existing Workflows

The caching system is fully backward compatible:
- Existing training scripts work without modification
- Cache is created automatically on first run
- No changes needed to model training code
- Works with all CDVAE configurations and datasets

## Future Enhancements

Potential improvements to the caching system:
- Compression for smaller cache files
- Distributed caching for multi-node training
- Cache warming utilities
- Cache analytics and optimization tools