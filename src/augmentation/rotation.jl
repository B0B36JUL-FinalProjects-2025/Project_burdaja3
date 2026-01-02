using ImageTransformations, Interpolations, Images

"""
    rotate!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, θ::Real)

Rotates a 3D Float32 image (H × W × C) by a given angle `θ` (in degrees) and writes
the result into the pre-allocated `out` buffer.

# Arguments
- `out::AbstractArray{Float32,3}` : pre-allocated output buffer of the same size as `img`.
- `img::AbstractArray{Float32,3}` : input image to rotate.
- `θ::Real` : rotation angle in degrees (can be positive or negative).

# Returns
- `out` : the rotated image (mutated in-place).
"""
function rotate!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, θ::Real)
    H, W, C = size(img)
    @assert size(out) == (H, W, C) "Output buffer must have the same size as the input."

    # Iterate over each channel
    for c in 1:C
        # Rotate the channel and write directly into the output buffer
        out[:,:,c] .= ImageTransformations.imrotate(
            img[:,:,c],          # input channel
            θ,                   # rotation angle
            axes(img[:,:,c]),    # maintain original axes
            method = BSpline(Linear()), # linear interpolation
            fillvalue = Reflect()        # reflect edges for padding
        )
    end

    return out
end
