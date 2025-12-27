using HDF5

"""
Load training datasets (one per label) as HDF5 dataset references.

Each returned element is an HDF5.Dataset pointing to the `"images"`
dataset inside:
    data/train/label_<label>.h5

Data are NOT loaded into memory until indexed.

# Arguments
- `path::String="data/train"`: directory containing per-label HDF5 files

# Returns
- `Vector{HDF5.Dataset}` of length 10 (one dataset per label)
"""
function load_train(path::String = "data/train")
    datasets = Vector{HDF5.Dataset}(undef, 10)

    for label in 0:9
        file_path = joinpath(path, "label_$(label).h5")

        @assert isfile(file_path) "Missing file: $file_path"

        file = h5open(file_path, "r")
        datasets[label + 1] = file["images"]
    end

    return datasets
end


"""
Load the test dataset into memory.

Reads:
    data/test/test.h5

# Arguments
- `path::String="data/test"`: directory containing the test dataset

# Returns
- `images::Array{UInt8,4}`  -- shape (H, W, C, N)
- `labels::Vector{Int}`
"""
function load_test(path::String = "data/test")
    file_path = joinpath(path, "test.h5")

    @assert isfile(file_path) "Missing file: $file_path"

    h5file = h5open(file_path, "r")

    images = Array(h5file["images"])
    labels = Array(h5file["labels"])

    close(h5file)

    return images, labels
end
