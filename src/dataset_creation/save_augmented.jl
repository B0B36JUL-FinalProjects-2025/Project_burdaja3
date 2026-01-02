using Random
using HDF5

include("../augmentation/augment.jl")

"""
    generate_augmented_images(img::Array{UInt8,3}; augments::Int=8)

Generate multiple augmented versions of a single image.
The output contains the original image followed by `augments`
randomly augmented variants.
# Arguments
- `img::Array{UInt8,3}`: input image of shape `(H, W, C)`
- `augments::Int=8`: number of augmented samples to generate
# Returns
- `Array{UInt8,4}` of shape `(H, W, C, augments + 1)`
"""
function generate_augmented_images(img::Array{UInt8,3}; augments::Int=8)
    # Convert a UInt8 image to Float32 in-place.
    @inline function u8_to_f32!(dst::Array{Float32,3}, src::Array{UInt8,3})
        @inbounds @simd for i in eachindex(src)
            dst[i] = Float32(src[i])
        end
        return dst
    end

    # Convert a Float32 image to UInt8 in-place with clamping.
    # Values are clipped to the range `[0, 255]` and rounded before conversion.
    @inline function f32_to_u8!(dst::Array{UInt8,3}, src::Array{Float32,3})
        @inbounds @simd for i in eachindex(src)
            v = src[i]
            v = ifelse(v < 0f0, 0f0, ifelse(v > 255f0, 255f0, v))
            dst[i] = UInt8(round(v))
        end
        return dst
    end

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
    save_augmented_dataset(filename::String,
                                dataset::Array{UInt8,4};
                                augments::Int=0,
                                key::String="images")

Save an augmented and shuffled version of a dataset into an HDF5 file.

Each input image is expanded into `(augments + 1)` samples
(original + augmented variants). All augmented samples are first stored
in memory, then randomly shuffled, and finally written to disk.

This ensures that later block-wise reading yields well-mixed data.

# Arguments
- `filename::String`: output HDF5 file path
- `dataset::Array{UInt8,4}`: input images `(H, W, C, N)`
- `augments::Int=0`: number of augmentations per image
- `key::String="images"`: dataset name inside HDF5
"""
function save_augmented_dataset(filename::String,
                                dataset::Array{UInt8,4};
                                augments::Int=0,
                                key::String="images")

    H, W, C, N = size(dataset)
    total_size = N * (augments + 1)

    # buffer for all augmented images (before shuffling)
    buffer = Array{UInt8}(undef, H, W, C, total_size)

    idx = 1
    for i in 1:N
        aug_data = generate_augmented_images(dataset[:,:,:,i], augments=augments)

        buffer[:,:,:,(idx):(idx + augments)] .= aug_data
        idx += augments + 1
    end

    # shuffle along sample dimension
    perm = randperm(total_size)
    buffer = buffer[:,:,:,perm]

    # write to HDF5
    h5open(filename, "w") do file
        file[key] = buffer
    end

    return nothing
end