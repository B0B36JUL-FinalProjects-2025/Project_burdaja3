using HDF5

"""
    load_galaxy(path::String = "data/Galaxy10_DECals.h5")

Load the original Galaxy10 dataset from the default HDF5 file.

The dataset is expected at:
    `data/Galaxy10_DECals.h5`

Images are reordered to shape `(H, W, C, N)`.

# Arguments
- `path::String="data/Galaxy10_DECals.h5"`: path to the Galaxy10 HDF5 file

# Returns
- `images::Array{UInt8,4}`: image tensor of shape `(H, W, C, N)`
- `labels::Vector{Int}`: corresponding class labels
"""
function load_galaxy(path::String = "data/Galaxy10_DECals.h5")
    @assert isfile(path) "Galaxy10 file not found: $path"

    h5file = h5open(path, "r")

    images = Array(h5file["images"])
    labels = Array(h5file["ans"])

    close(h5file)

    # reorder from (C, H, W, N) → (H, W, C, N)
    images = permutedims(images, (2, 3, 1, 4))

    return images, labels
end