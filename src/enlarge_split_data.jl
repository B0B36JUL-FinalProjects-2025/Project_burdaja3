using Random
using HDF5
include("augmentation/augment.jl")

"""
Convert a UInt8 image to Float32 in-place.

This function copies values from `src` to `dst` while converting
each pixel from `UInt8` to `Float32`.

Both arrays must have the same shape `(H, W, C)`.

# Arguments
- `dst::Array{Float32,3}`: destination buffer
- `src::Array{UInt8,3}`: source image

# Returns
- `dst`
"""
@inline function u8_to_f32!(dst::Array{Float32,3}, src::Array{UInt8,3})
    @inbounds @simd for i in eachindex(src)
        dst[i] = Float32(src[i])
    end
    return dst
end


"""
Convert a Float32 image to UInt8 in-place with clamping.

Values are clipped to the range `[0, 255]` and rounded before conversion.

# Arguments
- `dst::Array{UInt8,3}`: destination buffer
- `src::Array{Float32,3}`: source image

# Returns
- `dst`
"""
@inline function f32_to_u8!(dst::Array{UInt8,3}, src::Array{Float32,3})
    @inbounds @simd for i in eachindex(src)
        v = src[i]
        v = ifelse(v < 0f0, 0f0, ifelse(v > 255f0, 255f0, v))
        dst[i] = UInt8(round(v))
    end
    return dst
end


"""
Generate augmented versions of a single image.

The output contains the original image followed by `augments`
randomly augmented variants.

# Arguments
- `img::Array{UInt8,3}`: input image of shape `(H, W, C)`
- `augments::Int=8`: number of augmented samples to generate

# Returns
- `Array{UInt8,4}` of shape `(H, W, C, augments + 1)`
"""
function generate_augmented_images(img::Array{UInt8,3}; augments::Int=8)
    H, W, C = size(img)

    a = Array{UInt8}(undef, H, W, C, augments + 1)
    buffer_32 = Array{Float32}(undef, H, W, C)
    buffer_8 = Array{UInt8}(undef, H, W, C)

    # store original image
    a[:,:,:,1] = img

    # generate augmentations
    for i in 2:augments+1
        u8_to_f32!(buffer_32, img)

        augment!(buffer_32, buffer_32)

        f32_to_u8!(buffer_8, buffer_32)

        a[:,:,:,i] .= buffer_8

    end

    return a
end


"""
Save an augmented version of a dataset into an HDF5 file.

Each image in the input dataset is expanded into `(augments + 1)` images:
the original plus its augmented variants.

The resulting dataset is stored under the key `"images"`.

# Arguments
- `filename::String`: output HDF5 file path
- `dataset::Array{UInt8,4}`: input images `(H, W, C, N)`
- `augments::Int=8`: number of augmentations per image
"""
function save_augmented_dataset(filename::String,
                                dataset::Array{UInt8,4};
                                augments::Int=8)

    H, W, C, N = size(dataset)
    total_size = N * (augments + 1)

    h5open(filename, "w") do file
        file["images"] = zeros(UInt8, H, W, C, total_size)
        dset = file["images"]

        idx = 1
        for i in 1:N
            aug_data = generate_augmented_images(dataset[:,:,:,i], augments=augments)
            dset[:,:,:,(idx):(idx + augments)] = aug_data
            idx += augments + 1
        end
    end
end


"""
Load the Galaxy10 dataset from an HDF5 file.

The image tensor is reordered to shape `(H, W, C, N)`.

# Arguments
- `path::String`: path to the Galaxy10 HDF5 file

# Returns
- `images::Array{UInt8,4}`
- `labels::Vector`
"""
function load_galaxy(path::String)
    h5file = h5open(path, "r")

    images = Array(h5file["images"])
    labels = Array(h5file["ans"])

    close(h5file)

    # reorder to (H, W, C, N)
    images = permutedims(images, (2, 3, 1, 4))

    return images, labels
end


"""
Split dataset into training and test sets, augment training images,
and save them into HDF5 files.

For each class label:
- A fraction of images (`test_frac`) is saved as test set without augmentations
- Remaining images are augmented and saved as training set

# Arguments
- `path::String`: path to the original Galaxy10 dataset
- `test_frac::Float64=0.1`: fraction of images per class to use for test set
- `augments::Int=8`: number of augmentations per training image

# Output
- Train HDF5 files: `data/train/label_<label>.h5`
- Test HDF5 file: `data/test/test.h5` with `"images"` and `"labels"`
"""
function enlarge_split(path::String; test_frac=0.1, augments=8)
    images, labels = load_galaxy(path)

    # ensure output directories exist
    mkpath("data/train")
    mkpath("data/test")

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
        save_augmented_dataset("data/train/label_$(label).h5", train_images_label, augments=augments)
    end

    # concatenate test images along 4th dimension
    test_imgs = cat(test_imgs_list..., dims=4)
    # concatenate test labels into a single vector
    test_lbls = vcat(test_labels...)
    # save test images and labels
    h5open("data/test/test.h5", "w") do f
        f["images"] = test_imgs
        f["labels"] = test_lbls
    end

    return
end 
