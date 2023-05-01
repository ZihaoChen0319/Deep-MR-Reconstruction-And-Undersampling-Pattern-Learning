# Learn a Task-Adaptive MR Under-Sampling Pattern
A deep learning framework to learn a task-adaptive under-sampling pattern and reconstruct MRI in end-to-end way.

## Get Started

### Prerequisites
This project is based on python. Before starting playing with it, please ensure that you are familiar with python environment setup.

### Installation
1. Clone the repo
```
git clone https://github.com/ZihaoChen0319/Deep-MR-Reconstruction-And-Undersampling-Pattern-Learning
```

2.Install the required python packages
```
pip install -r requirements.txt
```

## Usage

### Prepare the dataset
The datasets used here are from [Medical Segmentation Decathlon](http://medicaldecathlon.com/), which contains 10 different medical image segmentation tasks. You can download the datasets by yourself or use your own datasets. 

Please place the dataset at `./Data` or the dir you like. The structure of dataset file folder should be like:
```
./Data
└── your_dataset
    ├── imagesTr
    ├── labelsTr
    └── (optional) dataset.json
```

### Data preprocessing
Please set `data_dir` and `data_name` in `data_preprocess.py` based on your situation, then run:
```
python data_preprocess.py
```
The processed data will be saved at `data_dir/data_name_np`.

### Training
Please set `if_train=True` and other parameters in `main.py` based on your situation, then run:
```
python main.py
```

### Evaluation
Please set `if_train=False` in `main.py` and run:
```
python main.py
```

## Roadmap
- [x] Implementation of the framework
- [ ] Add a framework figure
- [ ] Use `argparse` in `data_preprocess.py`, `main.py` and `visualize.py'

## Acknowledgement
### Open-Source Repositories
* [cagladbahadir/LOUPE](https://github.com/cagladbahadir/LOUPE)
* [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)

### References:
* C. D. Bahadir, A. Q. Wang, A. V. Dalca, and M. R. Sabuncu, “Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI,” IEEE Transactions on Computational Imaging, vol. 6, pp. 1139–1152, 2020
