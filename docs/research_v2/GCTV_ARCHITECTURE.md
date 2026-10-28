# GCTV architecture

G0: 12D observed input statistics, 32/64/96-channel dense 3D residual backbone, three128D ZYX FPN outputs. G1: adds XY/XZ/YZ observed/context/coverage plane projections and single residual lift at F2/F3. G2: also feeds the first voxel update back to refine each plane and lifts those refined planes into a second voxel update. Only partial LiDAR creates the observed mask. Outputs retain the original Stage1 and Stage2 feature interfaces.

Each output projection is independently zero initialized. This preserves initial backbone behavior, but plane/refinement parameters have zero gradients through that path until the corresponding output projection becomes nonzero; unit tests explicitly open the projection before verifying feedback and plane gradients. G2 with feedback disabled uses exactly G1's single-pass calculation when shared weights match. This is an implementation test, not evidence of a trained quality improvement.

No G3/G4/G5 or Stage2 modifications are registered. Compact3D G0/G1/G2 parameters, final speed-quality Pareto, complete numerical comparison and trained coupling benefit: NOT TESTED.
