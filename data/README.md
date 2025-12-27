# Galaxy10 Data Preparation

## 1. Download the dataset

Download the HDF5 file from:

[Galaxy10_DECals dataset](https://astronn.readthedocs.io/en/stable/galaxy10.html)

Place the file in the `data/` folder:

data/Galaxy10_DECals.h5

## 2. Generate augmented training data and split test set

Run the `enlarge_split` function from the `enlarge_split_data.jl` script:

```julia
include("src/enlarge_split_data.jl")

# Example usage
enlarge_split("data/Galaxy10_DECals.h5"; test_frac=0.1, augments=8)

### What happens

- The dataset is loaded from `data/Galaxy10_DECals.h5`.

- For each class label (0–9):
  - A fraction (`test_frac`) of images is reserved as **test set** and **not augmented**.
  - The remaining images are **augmented** (number of augmentations specified by `augments`) and saved as **train set**.

- The output is saved in HDF5 format:
  - **Training data per label**:
    ```
    data/train/label_0.h5
    data/train/label_1.h5
    ...
    data/train/label_9.h5
    ```
    containing:
    - `"images"` → test images

  - **Combined test set**:
    ```
    data/test/test.h5
    ```
    containing:
    - `"images"` → test images
    - `"labels"` → corresponding labels