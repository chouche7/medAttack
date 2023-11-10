# Understanding Adversarial Attacks on Deep Predictive Models for Electronic Medical Records
Based on the original paper [here](https://arxiv.org/abs/1802.04822). 

## Code Usage
- cw.py Algorithm for Generating Adversarial Examples
- cw_main.py Main functions
- Implementation was interagted from [EAD: Elastic-Net Attacks to Deep Neural Networks via Adversarial Examples](https://github.com/ysharma1126/EAD_Attack) and [Adversarial Algorithms in TensorFlow](https://zenodo.org/record/1154272#.W7TYc5NKhTY). 

These are the steps we took to reproduce the data:
∗ Dependencies: 
  - Python 3.7.8 
  - Tensorflow 1.13.0
  - Pandas 1.1.5
∗ Download instruction of data:
  - To download MIMIC3, follow the instructions under the "FILE" section of: https://physionet.org/content/mimiciii/1.4/
  - Make sure you unzip all the downloaded files or the code will not work.
∗ Functionality of scripts: preprocessing, training, evaluation, etc.
  - data_extraction.py: Reads in original MIMIC3 csvs and extracts only the necessary columns (vital signs, lab tests, mortality), then writes it back into smaller files for processing (some of the the original files contain over 500 million rows which makes it difficult to process)
  - data_transformation.py: Reads in the simplified files and conducts pre-processing which includes imputation, outlier removal, normalization, and padding of timeseries. We also split the dataset into multiple folds with downsampling involved in order to balance the classes.
  - cw.py: Original author code that generates adversarial attack
  - cw_main.py: Original author code that trains the 3-layer LSTM model. 
∗ Instruction to run the code
  1. Have all the dependencies ready.
  2. Download the MIMIC3 dataset.
  3. Unzip all files.
  4. Run data_extraction.py.
  5. Run data_transformation.py.
  6. Run cw_main.py.
