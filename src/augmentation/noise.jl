using Random, Images

"""
    noise!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, mean::Float32=0.0, std::Float32=0.05)

Adds Gaussian noise to a 3D image (H × W × C) and writes the result into the pre-allocated `out` buffer. 

# Arguments
- `out::AbstractArray{Float32,3}` : pre-allocated output buffer of the same size as `img`.
- `img::AbstractArray{Float32,3}` : input image (H × W × C).

Keyword arguments
- `mean::Float32` : mean of the Gaussian noise (default is 0.0).
- `stddev::Float32` : standard deviation of the Gaussian noise.

# Returns
- `out` : image with added noise (mutated in-place).
"""
function noise!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}; mean::Float32=0.0f0, stddev::Float32=0.05f0)
    H, W, C = size(img)
    @assert size(out) == (H, W, C) "Output buffer must have the same size as the input."

    # Generate noise for all channels simultaneously
    noise = randn(Float32, H, W, C) * stddev .+ mean  # Noise for all channels (R, G, B)

    # Add the noise
    out .= img .+ noise

    return out
end
