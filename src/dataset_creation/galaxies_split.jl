using Random
using HDF5

include("load_galaxy.jl")
include("save_augmented.jl")

"""
    split(;path::String = "data/Galaxy10_DECals.h5", test_frac=0.1, augments=0)

Split dataset into training and test sets, augment training images,
and save them into HDF5 files by their labels.

For each class label:
- A fraction of images (`test_frac`) is saved as test set without augmentations
- Remaining images are augmented and saved as training sets

# Arguments
- `path_in::String`: path to the original Galaxy10 dataset
- `path_out::String`: path to the output dataset directories
- `test_frac::Float64=0.1`: fraction of images per class to use for test set
- `augments::Int=0`: number of augmentations per training image

# Output
- Train HDF5 files: `data/train/label_<label>.h5`
- Test HDF5 file: `data/test/test.h5` with `"images"` and `"labels"`
"""
function split(;path_in::String = "data/Galaxy10_DECals.h5", path_out::String = "data", test_frac=0.1, augments=0)
    images, labels = load_galaxy(path_in)

    # ensure output directories exist
    train_dir = joinpath(path_out, "train")
    test_dir  = joinpath(path_out, "test")
    mkpath(train_dir)
    mkpath(test_dir)

    test_imgs_list  = []  # collect test images per label
    test_labels     = []  # collect corresponding labels

    for label in 0:9
        idx = findall(labels .== label)
        N = length(idx)
        n_test = round(Int, test_frac * N)

        shuffled = shuffle(idx)

        # select test and train indices
        test_idx = shuffled[1:n_test]
        train_idx = shuffled[n_test+1:end]

        # store test images and labels
        push!(test_imgs_list, images[:,:,:,test_idx])
        push!(test_labels, fill(label, length(test_idx)))

        # augment and save train images per label
        train_images_label = images[:,:,:,train_idx]
        save_augmented_dataset(joinpath(train_dir, "label_$(label).h5"), train_images_label, augments=augments)
    end

    # concatenate test images along 4th dimension
    test_imgs = cat(test_imgs_list..., dims=4)
    # concatenate test labels into a single vector
    test_lbls = vcat(test_labels...)
    # save test images and labels
    h5open(joinpath(test_dir, "test.h5"), "w") do f
        f["images"] = test_imgs
        f["labels"] = test_lbls
    end

    return
end 
