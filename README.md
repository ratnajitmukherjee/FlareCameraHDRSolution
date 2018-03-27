# FlareCameraHDRSolution
Solution to extract (debayer) multiple exposures from Flare (IO Industries) cameras and merge them into HDRs. 

PLEASE NOTE:
1) A part of this work is parallelized using joblib and multiprocessing i.e. "the debayer" process. Tested on an i9 (10 core - 20 logical) CPU offering a massive speedup in the debayering process.

2) Another technique of debayering includes usage of NVIDIA gpus. However, the speedup offered by the parallelization is sufficient in this case. Therefore, this is ignored until further notice.
