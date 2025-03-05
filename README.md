# Flexible Secure Biometrics Hashing Network (FSB-HashNet)
## Flexible Secure Biometrics: A Protected Modality-Invariant Face-Periocular Recognition System

![Network Architecture](FSB_HashNet.png?raw=true "FSB-HashNet")

The project directories are as follows:

- analysis_privacy_security: Contains evaluation for security and privacy analysis - Non-Invertibility, Revocability, Unlinkability.
    * evaluateNonInvertibility - Evaluate non-invertibility criteria via genuine-impostor graph.
    * evaluateRevocability - Evaluate revocability criteria, only can be used after executing `generate_scores.py`.
    * evaluateUnlinkability - Evaluate unlinkability criteria, only can be used after executing `generate_scores.py`.
    * generate_scores - Generate genuine-impostor and mated-non-mated scores that are used for revocability and unlinkability evaluation.
    * noninvert_calculate_sar - Evaluate non-invertibility criteria by calculating the Success Attack Rate (SAR).
- configs: Dataset path configuration file and hyperparameters.
    * datasets_config.py - Directory path for dataset files. Change 'main' in 'main_path' dictionary to point to dataset, e.g., `/home/tiongsik/fsb_hashnet/data` (without slash).
    * params.py - Adjust hyperparameters and arguments in this file for training. 
- data: Dataloader functions and preprocessing.
    * __**INSERT DATASET HERE**__
    * _ROC data dictionaries are generated in this directory._
    * data_loader.py - Generate training and testing PyTorch dataloader. Adjust the augmentations etc. in this file. Batch size of data is also determined here, based on the values set in `params.py`.
- eval: Evaluation metrics (identification and verification). Also contains CMC and ROC evaluations.
    * plot_roc_sota.ipynb - Notebook to plot ROC curves, based on generated `.pt` files from `verification.py`. Graph is generated in `graphs` directory.
    * verification.py - Evaluates Verification Equal Error Rate (EER) and generates Receiver Operating Characteristic (ROC) curve, which are saved as `.pt` files in `data` directory. Use these `.pt` files to generate ROC curves.
- graphs: Directory where graphs are generated.
    * _ROC curve file is generated in this directory._
- logs: Directory where logs are generated.
    * _Logs will be generated in this directory. Each log folder will contain backups of training files with network files and hyperparameters used._
- models: Directory to store pre-trained models. Trained models are also generated in this directory.
    * __**INSERT PRE-TRAINED MODELS HERE. The base MobileFaceNet for fine-tuning the FSB-HashNet can be downloaded in [this link](https://www.dropbox.com/scl/fo/zqc0b6qw04189onq3b1as/AAt0D5d-HlV5OW8r2QPHB4k?rlkey=p0t8nqu8zixg2ibfhlvvr8t0y&st=nodkhbxh&dl=0).**__
    * _Trained models will also be stored in this directory._
- network: Contains loss functions and network related files.
    * fsb_hash_net.py - Architecture file for FSB-HashNet. Also used in the case of hashed (stolen token / user-specific token).
    * fsb_hash_net_baseline - Architecture file for baseline FSB-HashNet. Also used in the case of non-hashed baseline.
    * load_model.py - Loads pre-trained weights based on a given model.
    * logits.py - Contains some loss functions that are used.
- __training:__ Main files for training FSB-HashNet.
    * main.py - Main file to run for training. Settings and hyperparameters are based on the files in `configs` directory.
    * train.py - Training file that is called from `main.py`. Gets batch of dataloader and contains criterion for loss back-propagation.

### Pre-requisites (requirements):
Check `requirements.txt` file which was generated using `pip list --format=freeze > requirements.txt` command. These files are slightly filtered, so there may be redundant packages.
Download dataset (training and testing) from [this link](https://www.dropbox.com/s/bfub8fmc44tvcxb/periocular_face_dataset.zip?dl=0). Password is _conditional\_biometrics_.

### Training:
1. Ensure that datasets are located in `data` directory. Configure `datasets_config.py` file to point to this data directory.
2. Change hyperparameters accordingly in `params.py` file. The set values used are the default.
3. Run `main.py` file. The training should start immediately.
4. Testing will be performed automatically after training is done, but it is possible to perform testing on an already trained model (see next section).

### Testing:
0. Pre-trained models for fine-tuning or testing can be downloaded from [this link](https://www.dropbox.com/s/g8gn4x4wp0svyx5/pretrained_models.zip?dl=0). Password is _conditional\_biometrics_.
1. Based on the (pre-)trained models in the `models(/pretrained)` directory, load the correct model and the architecture (in `network` directory) using `load_model.py` file. Change the file accordingly in case of different layer names, etc. 
2. Evaluation:
    * Verification / Receiver Operating Characteristic (ROC) curve: Run `verification.py`. Based on the generated `.pt` files in `data` directory, run `plot_roc_sota.ipynb` to generate ROC graph. For cases of stolen-token scenario, and user-specific token scenario, refer to the main function in `verification.py` for the configuration.
    * Security and Privacy Analysis: Under the `analysis_privacy_security` folder, run the desired criterion for desired output. In the case of revocability and unlinkability, `generate_scores.py` has to be run first. Whereas for non-invertibility, `evaluateNonInvertibility.py` generates the genuine-impostor graph, while `noninvert_calculate_sar.py` calculates the Success Attack Rate for non-invertibility. Graphs for privacy evaluation will be generated in this folder.

### Comparison with State-of-the-Art (SOTA) models

| Method | Intra-Modal EER (%) <br> (Periocular) | Intra-Modal EER (%) <br> (Face) | Cross-Modal EER (%) <br> (Periocular-Face) |
| --- | --- | --- | --- |
| CB-Net [(Paper)](https://ieeexplore.ieee.org/abstract/document/10201879) | 9.62 | 3.21 | 9.80 |
| HA-ViT [(Paper)](https://ieeexplore.ieee.org/document/10068230) | 11.39 | 10.29 | 13.14 |
| GC<sup>2</sup>SA-Net [(Paper)](https://ieeexplore.ieee.org/document/10418204) | 6.39 | 3.14 | 6.50 |
| MFA-ViT [(Paper)](https://ieeexplore.ieee.org/document/10656777) | 11.18 | 5.17 | 9.41 |
| [FSB-HashNet (No Hash)](https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0) | 7.10 | 3.26 | 6.77 |
| [FSB-HashNet (Stolen-Key)](https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0) | 9.18 | 4.75 | 9.77 |
| [FSB-HashNet (User-Specific)](https://www.dropbox.com/scl/fi/4ifobdj33k43xv45w9zjw/fsb_hashnet.zip?rlkey=i8gpmg9n6vezsofkciuk2zboe&st=icyz1nd5&dl=0) | 4.24 | 2.11 | 5.26 |