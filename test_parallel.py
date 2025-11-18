#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDAnalysis as mda
import maicos

mda.start_logging()

if __name__ == "__main__":
    u = mda.Universe("run.tpr", "run.xtc", in_memory=True)
    len(u.trajectory)


    ana_obj = maicos.DielectricPlanar(u.atoms, unwrap=True)
    ana_obj2 = maicos.DielectricPlanar(u.atoms, unwrap=False)


    ana_collection = maicos.core.AnalysisCollection(*[ana_obj, ana_obj2])

    ana_collection.run(verbose=False, n_workers=2, backend='dask', step=10)
    #ana_collection.run(verbose=False, n_workers=8, backend='serial', step=10)

    ana_obj2.n_frames
    print(ana_obj.frames)
    print(ana_obj._obs)
    print(ana_obj.means)

    #print(ana_obj._obs)
    #print(ana_obj.results.eps_perp)

