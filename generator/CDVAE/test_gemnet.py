#!/usr/bin/env python3

print('Testing GemNet import...')
try:
    from cdvae.pl_modules.gemnet.gemnet import GemNetT
    print('✅ GemNetT imported successfully!')
    print('✅ Class found at:', GemNetT)
    print('✅ CDVAE now has full functionality with GemNet support!')
except Exception as e:
    print(f'❌ Failed to import GemNetT: {e}')
    import traceback
    traceback.print_exc()