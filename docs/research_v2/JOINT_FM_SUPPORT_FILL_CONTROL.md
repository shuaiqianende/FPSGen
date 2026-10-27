# Same-support Stage1 fill versus Stage2 refinement

41 frames, seq08/skip100, native Fine voxel centers, Stage1 ordinary FM NFE2/threshold0.5.
All41 Parent-ID hashes match both joint-inference arms at every NFE. No extra or GT-selected Parents.
The fill control occupies every valid Child inside the predicted Parents; it is not a learned Fine decoder.
Stage2 rows use NFE4 and the historical25%-augmentation-trained weights, which remain noncompliant training results.

|Output|Native Fine IoU %|P %|R %|CD m|IoU0.5 %|IoU0.2 %|IoU0.1 %|JSD-BEV|JSD-3D|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Stage1 support filled with all valid Children|29.7324|39.9154|54.9718|0.244019|47.9998|36.3478|14.8800|0.326263|0.449763|
|no_encoder|25.6717|60.4034|31.2170|0.295027|42.8488|31.6180|12.2027|0.365869|0.459810|
|with_encoder|26.0550|60.3746|31.7937|0.296525|42.7127|31.8634|12.3673|0.366346|0.459504|

Macro averages over41 fixed frames. Different occupancy densities can trade off native IoU, CD and physical IoU; report them together. No latency claim or compliant retraining is implied.
