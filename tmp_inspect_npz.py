import os, glob, numpy as np
paths = sorted(glob.glob('src/multiGPU/results/aggregate/dem_local_r000_*.npz'))
print('Found', len(paths), 'files')
for p in paths:
    st = os.stat(p)
    print('\nFILE:', p)
    print('  size(bytes):', st.st_size)
    try:
        with np.load(p) as d:
            for k in d.files:
                a = d[k]
                print('  key:', k, 'shape:', getattr(a, 'shape', None), 'dtype:', getattr(a, 'dtype', None))
    except Exception as e:
        print('  Failed to read npz:', e)

# Compare to dem_all files
paths_all = sorted(glob.glob('src/multiGPU/results/aggregate/dem_all_*.npz'))
print('\nFound dem_all:', len(paths_all))
for p in paths_all:
    st = os.stat(p)
    print('\nFILE:', p)
    print('  size(bytes):', st.st_size)
    try:
        with np.load(p) as d:
            for k in d.files:
                a = d[k]
                print('  key:', k, 'shape:', getattr(a, 'shape', None), 'dtype:', getattr(a, 'dtype', None))
    except Exception as e:
        print('  Failed to read npz:', e)
