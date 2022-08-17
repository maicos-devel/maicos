==============
General design
==============

----------
Foundation
----------

.. Show a flow chart here

All MAICoS analysis modules build on top of several stacked base classes which 
for positional analysis are 
split into the geometries: planar, cylinder and sphere. Each sub class 
inherits attributes and provides geometry specific methods and attributes. 
The flow chart is shown in the figure above.
The foundation for all these classes is
`AnalysisBase`, inherited and extended from 
`MDAnalysis.analysis.AnalysisBase`. `AnalysisBase` provides is caring 
about the general aspects of each analysis which will 
discussed in details below

1. **Atom Selection**
   Since MAICoS builds on top of the MDAnalysis Universe and atom selection 
   system the analysis all analysis modules are able to work only on subsets 
   of the whole simulation. This allows investigating different species components  
   individually for example splitting solvent and solute contribution to an 
   observable. Moreover, many MAICoS analysis modules 
   are able to process several atom selections from 
   one simulation within one analysis run by providing a `list` os atom selections.
   This reduces I/O loads and operations 
   and gains a speed up for the analysis. 

2. **Translational coordinate transformations and unit cell wrapping**
   MAICoS works with a reference structure denoted by `refgroup` 
   which center of mass (com) serves as the coordinate origin for 
   every analysis. MDAnalysis's cell dimension and coordinates range from 
   `0` to `L` where 
   `L` is the dimension of the simulation box. Therefore, MAICoS defines the 
   origin at the center of the simulation cell.
   
   Within each frame of the analysis the `refgroup`'s com 
   is translated to the origin and all coordinates are wrapped into the 
   primary unit cell. Additionally, it is possible to unwrap molecules afterwords
   since especially dielectric analysis require whole molecules. With this 
   centering it easily possible to investigate systems that translate over time 
   like soft interfaces or moving molecules. 
   However, users are not forced to give a `refgroup`. If no such 
   reference structure is given MAICoS takes the frame specific center 
   of the simulation cell as the origin.

   User provided ranges for spatial analysis are always with respect to the 
   `refgroup` and not in abolsute box coordinates. 
   For example a 1-dimensional planar analysis ranging from `-2 (Å)` to `0` 
   considers atoms on the left half space of the `refgroup`.

3. **Trajectory iteration**
   Each module only has to implement an initialization, prepare, single frame and a conclude 
   method. The `AnalysisBase` will based on these provided methods perform the analysis. 
   Of course it is possible to provide an initial and final frame as well as a step size. 
   The analysis of individual frames is also possible.

4. **Time averaging of observables**
   For observables that has to be time averaged `AnalysiBase` provides a Frame dictionary.
   Each key has to be updated within the `single_frame` method and the mean and 
   the variance of each observables will be provided within a `mean` and a `var` 
   dictionary. Each key name within these two dictionaries is the same as within the 
   frame dictionary.

5. **On-the-fly output**
   Due to the provided dictionaries and the discussed in the section before MAICoS is
   able to update analysis results during the analysis. This can be in particular useful 
   for long analysis providing a way to check the correctness of analysis parameters 
   during the run.

6. **Correlation time estimation**
   For the calculation of the mean and the standard deviation MAICoS assumes 
   uncorrelated data. Since users may not know the correlation time within their 
   simulation MAICoS will estimate correlation times and warns users if their 
   averages are obtained from correlated data. For dielectric analysis MAICoS 
   uses the total dipole moment perpendicular in the direction of the analysis. 
   For other spatial dependant analysis the correlation time is estimated 
   from the central bin of the refgroup; in the center of the simulation cell.

--------------------------
Spatial Dependent Analysis
--------------------------

A spatial dependent analysis is crucial for interfacial and systems in 
confinement. Based on the `AnalysisBase` MAICoS provides intermediate 
base classes 
for the three main geometries: PlanarBase, CylinderBase and SphereBase.
These modules will take of the coordinate transformations as well as the 
spatial boundaries as well as the spatial resolution of the analysis. 

A design concept of MAICoS for spatial analysis is that the user 
always provides the spatial resolution 
via the `binwidth` parameter rather than a number of bins. With this the same analysis 
code is easily transferable to different simulation size without additional 
considerations about the spatial resolution.

Based on the three geometric base classes three corresponding 
high level classes ProfilePlanarBase, ProfileCylinderBase and ProfileSphereBase
are provided. For developing a Analysis class based on theses three only a 
single function calculating the atomic weights has to be provided. 
atomic weights could be masses resulting in mass density profiles as done 
in `DensityPlanar`, atomic or molecular velocities as for `VelocityPlanar` or 
even dipolar orientations as used by the `Diporder` class. 
The profile 
base classes will based on the weighting function provide spatial 
at user requested spatial resolution averaged over variable part of the trajectory.

More details on each base class is given in the API Documentation and for
detailed information on the physical principles of each module consider 
the following sections.
