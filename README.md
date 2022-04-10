# Lung Annotator
An application to train and predict 3D lung point cloud annotations using a deep learning method.

Requirements
------------

* Python 3.9
* PyTorch
* tqdm
* Scikit-learn
* NumPy

For GPU implementation, you will need an NVIDIA driver, CUDA (v11.2 used to test this), cuDNN (compatible
version with CUDA and PyTorch).

-- Optional: for a quick visualisation, you will need a C++ compiler to build the visualisation script:

    cd scripts
    bash build.sh  # this will compile and build the 'utils/render.cpp' visualisation code.

Installation
------------
Lung Annotator can simply be installed by:

    pip install lung-annotator

Or, if you'd like to contribute to this project, you can clone this repository and in the top-level directory of 
the cloned repository run `poetry install`. (You can install `poetry` using `pip install poetry`).


---
**NOTE**

For a smooth installation, you can create a Python virtual environment within the ABI's `hpc5`. 

    virtualenv -p python3.9 <venv-name>
    source <venv-name/bin/activate>
    pip install lung-annotator

---


Training
------------
Simply modify the configuration file in `tests/resources/config` or create your own, and run:

    cd src
    python train.py <path/to/config>

After every epoch, a model checkpoint file `.pth` is saved with all the model states to use for prediction.
You can set the output directory inside the `confi` file (`output_model` field). 

Predict
------------
To predict the annotations of a new subject's lung 3D point data, you first need to convert the `.exnode`
files to `.pts` files. A simple script has been created for this task in this [GitHub repository](https://github.com/mahyar-osn/lung-point-generator).

You can then run the following to predict the annotations for your new data:

    python predict.py --input_file <path/to/.pts/file> --model <paht/to/model/checkpoint> --feature 1

If you have trained your model with the `feature_transform` option set to `True`, you will need the 
`feature` argument in the prediction set to `1` as shown above. Otherwise, set `--feature` to `0`.