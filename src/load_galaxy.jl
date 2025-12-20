using HDF5

"""
    load_galaxy(path::String)

Load the Galaxy10_DECals dataset from an HDF5 file.

# Arguments
- `path::String` : path to the HDF5 file (e.g., "data/Galaxy10_DECals.h5")

# Returns
- `images::Array{UInt8,4}` : array of images (256, 256, 3, N)
- `labels::Vector{Int}` : corresponding galaxy labels
"""
function load_galaxy(path::String;)
    # Open the HDF5 filee
    h5file = h5open(path, "r")


    # Read images and labels
    images = Array(h5file["images"])
    labels = Array(h5file["ans"])
    
    # Close the file
    close(h5file)

    images = permutedims(images, (2, 3, 1, 4))
    
    return images, labels
end