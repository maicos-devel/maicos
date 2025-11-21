#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDAnalysis as mda
import maicos

mda.start_logging()

if __name__ == "__main__":
    #u = mda.Universe("tests/data/water/water.gro", "tests/data/water/water.trr", in_memory=True)
    u = mda.Universe("run.tpr", "run.xtc", in_memory=True)
    len(u.trajectory)

    dens = maicos.DensityPlanar(u.atoms, jitter=0.0)
    dens.run(verbose=True, step=10, backend="serial", n_workers=1)
    #dens_jitter = maicos.DensityPlanar(u.atoms, bin_width=1e-6, jitter=0.01).run(step=10, verbose=True)


    ana_obj = maicos.DielectricPlanar(u.atoms, unwrap=True)
    ana_obj2 = maicos.DielectricPlanar(u.atoms, unwrap=False)


    ana_collection = maicos.core.AnalysisCollection(*[ana_obj])

    #ana_collection.run(verbose=False, n_workers=2, backend='dask', step=10)
    ana_collection.run(verbose=True, n_workers=1, backend='serial', step=10)

    print(ana_obj.frames)
    print(ana_obj._obs)
    print(ana_obj.means)

    #print(ana_obj._obs)
    #print(ana_obj.results.eps_perp)

