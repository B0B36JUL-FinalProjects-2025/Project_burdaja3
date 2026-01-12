# Galaxy10 Data Preparation

## 1. Download the dataset

Download the HDF5 file from:

[Galaxy10_DECals dataset](https://astronn.readthedocs.io/en/stable/galaxy10.html)

Place the file in the `data` folder:

data/Galaxy10_DECals.h5

## 2. Generate augmented training datasets and split test set

Run the `split` function:

```julia
split(path_in="data/Galaxy10_DECals.h5", path_out="data", test_frac=0.1, augments=0)
```


- The dataset is loaded from **path_in**.

- For each class label (0–9):
  - A fraction (`test_frac`) of images is reserved as **test set** and **not augmented**.
  - The remaining images are **augmented** (number of augmentations specified by `augments`) and saved as **train set**.
  - Augmentation can either happen before training or during each training batch. Augmenting during each batch is typically more effective for generalization, while pre-augmenting is faster.

- The output is saved in HDF5 format:
  - **Training data per label**:
    ```
    path_out/train/label_0.h5
    path_out/train/label_1.h5
    ...
    path_out/train/label_9.h5
    ```
    containing:
    - `"images"` → train images

  - **Combined test set**:
    ```
    path_out/test/test.h5
    ```
    containing:
    - `"images"` → test images
    - `"labels"` → corresponding labels