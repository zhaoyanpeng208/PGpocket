# PGpocket
This is the implementation of PGpocket: : A point cloud graph neural network for protein-ligand binding site prediction

## catalogue

- [Installation](#Installation)
- [Dataset](#Dataset)
- [Train](#Train)
- [Prediction](#Prediction)
- [License](#License)


## Installation
PGpocket is built on Python3, we recommend using a virtual conda
 environment as enviroment management for the installation of 
 PGpocket and its dependencies. 
 The virtual environment can be created as follows:
```bash
conda create -n your_environment python==3.9
conda activate your_environment
```
Download the source code of PGpocket from GitHub:
```bash
git clone https://github.com/username/my-project.git
```
Install PGpocket dependencies as following:
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cu102 -c pytorch
pip install torch-geometric==2.4.0
pip install torch-cluster==1.6.3
pip install torch-scatter==2.1.2
pip install torch-sparse==0.6.18
pip install torch-spline-conv==1.2.2
pip install pykeops==2.1.2
pip install plyfile==1.0.2
pip install pytz==2023.3.post1
```
## Dataset
The PDB sturcture used in this study can be download from the link 
https://example.com
## Train
Multiple hyperparameters can be selected in main.py.
```bash
python main.py
```
After model training starts, the progress bar will be automatically shown on your command line， and the trained model parameters will be saved in "runs" dictory for every epoch.
## Prediction
Model parameters can be found under the runs folder
```bash
python predict.py
```
After predicting with your well trained model, the predicting output will be saved in a "npy" file.
## License
This project is covered under the Apache 2.0 License.

