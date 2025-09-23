import glob, os, datetime
files = sorted(glob.glob('src/multiGPU/results/aggregate/*.npz'))
for p in files:
    mtime = os.path.getmtime(p)
    print(p, 'mtime:', datetime.datetime.fromtimestamp(mtime))
