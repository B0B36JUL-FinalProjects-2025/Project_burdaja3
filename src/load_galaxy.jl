using HDF5

"""
    load_galaxy(path::String)

Load the Galaxy10_DECals dataset from an HDF5 file.

# Arguments
- `path::String` : path to the HDF5 file (e.g., "data/Galaxy10_DECals.h5")

# Returns
- `images::Array{UInt8,4}` : array of images (N, 3, 256, 256)
- `labels::Vector{Int}` : corresponding galaxy labels
"""
function load_galaxy(path::String)
    # Open the HDF5 filee
    h5file = h5open(path, "r")
    
    # Read images and labels
    images = read(h5file["images"])
    labels = read(h5file["ans"])
    
    # Close the file
    close(h5file)

    images = permutedims(images, (4,1,2,3))
    
    return images, labels
end