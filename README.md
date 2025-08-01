# MultipleReservoirComputing-MRC
This repository contains proposed methods for Sign Language Recognition (SLR) using multiple reservoir computing (MRC).

## Steps to Run the Code

1. Set the directory in each code, then run the program in the order as below.
2. `LoadData.py` has a function to copy video datasets from the WLASL dataset into the target folder
3. `ExtractTheKeypoint.py` is used to extract the keypoint from the video dataset and write on CSV the video name, total extracted frames and the action labels
4. `LoadExtractedKeypoint.py` has a function to load the data and save into `.npy`
5. Run the classification algorithm such as `Sign_language_Conv1D_BiGRU.py`, `Sign_language_BiGRUDropout.py`, `ReservoirRewriteMulti_withoutOptuna.py`, etc

## How to Access the Dataset

Please access the dataset from https://dxli94.github.io/WLASL/

# Citation
If you find this work useful, please cite it as follows:

Syulistyo A, Tanaka Y, Pramanta D, Fuengfusin N, Tamukoh H (2025) Low-cost computation for isolated sign language video recognition with multiple reservoir computing. PLoS One 20(7): e0322717. https://doi.org/10.1371/journal.pone.0322717
