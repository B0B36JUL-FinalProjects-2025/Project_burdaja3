using Random, Images

"""
    noise!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, mean::Float32=0.0, stddev_perc::Float32=0.05, max_value::Float32=1.0)

Adds Gaussian noise (with a given `mean` and `stddev_perc` as a percentage of `max_value`) to a 3D image (H × W × C) and writes the result into the pre-allocated `out` buffer. 
The pixel values are clamped between 0 and a specified `max_value` (default is 1.0, suitable for normalized images).

# Arguments
- `out::AbstractArray{Float32,3}` : pre-allocated output buffer of the same size as `img`.
- `img::AbstractArray{Float32,3}` : input image (H × W × C).

Keyword arguments
- `mean::Float32` : mean of the Gaussian noise (default is 0.0).
- `stddev_perc::Float32` : standard deviation of the Gaussian noise as a percentage of `max_value` (default is 0.05, or 5%).
- `max_value::Float32` : the maximum value for pixel intensities (default is 1.0, suitable for images normalized between 0 and 1).

# Returns
- `out` : image with added noise (mutated in-place).
"""
function noise!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}; mean::Float32=0.0f0, stddev_perc::Float32=0.05f0, max_value::Float32=1.0f0)
    H, W, C = size(img)
    @assert size(out) == (H, W, C) "Output buffer must have the same size as the input."

    stddev = stddev_perc * max_value

    # Generate noise for all channels simultaneously
    noise = randn(Float32, H, W, C) * stddev .+ mean  # Noise for all channels (R, G, B)

    # Add the noise and clamp the values between 0 and max_value
    out .= clamp(img .+ noise, 0.0f0, max_value)

    return out
end
