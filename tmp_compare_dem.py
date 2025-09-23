import glob, os, numpy as np

def inspect():
    dem_all = sorted(glob.glob('src/multiGPU/results/aggregate/dem_all_*.npz'))
    if not dem_all:
        print('No dem_all files found')
        return
    for allp in dem_all:
        inbase = os.path.splitext(os.path.basename(allp))[0].replace('dem_all_','')
        print('\n=== INPUT:', inbase)
        st = os.stat(allp)
        print('dem_all file:', allp)
        print('  size(bytes):', st.st_size)
        with np.load(allp) as d:
            for k in d.files:
                a = d[k]
                print('  key:', k, 'shape:', getattr(a,'shape',None), 'dtype:', getattr(a,'dtype',None))
        # find matching dem_local files
        local_pattern = f'src/multiGPU/results/aggregate/dem_local_r*_{inbase}.npz'
        locals = sorted(glob.glob(local_pattern))
        print('Found', len(locals), 'per-rank local files:')
        total_rows = 0
        for p in locals:
            st = os.stat(p)
            print('  ', p, 'size:', st.st_size)
            with np.load(p) as d:
                # assume key is dem_local
                if 'dem_local' in d.files:
                    a = d['dem_local']
                    print('    dem_local shape:', a.shape)
                    total_rows += a.shape[0]
                else:
                    print('    Unexpected keys:', d.files)
        # sanity compare
        with np.load(allp) as d:
            main_rows = None
            if 'dem_all' in d.files:
                main_rows = d['dem_all'].shape[0]
            elif 'dem_local' in d.files:
                main_rows = d['dem_local'].shape[0]
            else:
                print('  dem_all file has unexpected keys', d.files)
        print('Sum of per-rank rows:', total_rows)
        print('Rows in dem_all:', main_rows)
        if total_rows == main_rows:
            print('OK: per-rank pieces sum to dem_all')
        else:
            print('MISMATCH: per-rank sum != dem_all (possible missing ranks or write failure)')

if __name__ == "__main__":
    inspect()
