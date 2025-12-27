using Flux: onehotbatch
using Random
using HDF5

"""
Load a random contiguous block from a training HDF5 dataset.

The dataset is assumed to have shape `(H, W, C, N)` and to already be
shuffled on disk.

# Arguments
- `dset::HDF5.Dataset`: dataset handle (e.g. one element returned by `load_train`)
- `block_size::Int`: number of samples to load

# Returns
- `Array{UInt8,4}` of shape `(H, W, C, block_size)`
"""
function load_random_block(dset::HDF5.Dataset, block_size::Int)
    H, W, C, N = size(dset)  # get number of samples from dataset
    start = rand(1:(N - block_size + 1))
    inds = start:(start + block_size - 1)

    return dset[:,:,:,inds]
end




"""
Load a full batch from all training datasets (one per label).
For each class, a random contiguous block of size `block_size` is loaded,
then all blocks are concatenated into a pre-allocated buffer.  
Labels are generated as one-hot vectors.  
# Arguments
- `train_sets::Vector{HDF5.Dataset}`: vector of HDF5.Dataset, one per class (from `load_train()`)
- `block_size::Int`: number of samples per class to include in the batch
# Returns
- `x::Array{Float32,4}`: images normalized to [0,1], shape `(H, W, C, num_classes * block_size)`
- `y::Array{Bool,2}`: one-hot labels, shape `(num_classes, num_classes * block_size)`
"""
function load_batch(train_sets::Vector{HDF5.Dataset}, block_size::Int)
    num_classes = length(train_sets)
    H, W, C, _ = size(train_sets[1])
    total_size = num_classes * block_size

    # Pre-allocate final buffer for images and labels
    x_uint8 = Array{UInt8}(undef, H, W, C, total_size)
    labels_list = Vector{Int}(undef, total_size)

    # Load a random block from each class and fill the buffer
    for (i, dset) in enumerate(train_sets)
        start_idx = (i-1)*block_size + 1
        end_idx   = i*block_size

        # Load a random contiguous block from this class
        block = load_random_block(dset, block_size)

        # Assign block into pre-allocated buffer
        x_uint8[:,:,:,start_idx:end_idx] .= block
        labels_list[start_idx:end_idx] .= i-1  # 0-based labels
    end

    # Normalize images to Float32 [0,1]
    x = (Float32.(x_uint8)) ./ 255f0

    # Convert labels to one-hot encoding
    y = onehotbatch(labels_list, 0:num_classes-1)

    return x, y
end

