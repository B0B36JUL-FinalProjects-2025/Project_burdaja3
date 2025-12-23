"""
    translate!(out::Array{T,3}, img::Array{T,3}, dx::Int, dy::Int) where T

Translates a 3D image (H × W × C) by integer offsets `(dx, dy)` and writes
the result into the pre-allocated `out` buffer. Areas outside the original
image are filled using reflection.

# Arguments
- `out::Array{T,3}` : pre-allocated output buffer of the same size as `img`.
- `img::Array{T,3}` : input image to translate.
- `dx::Int` : translation along the x-axis (rows). Positive shifts down.
- `dy::Int` : translation along the y-axis (columns). Positive shifts right.

# Returns
- `out` : the translated image (mutated in-place).

# Behavior
- Each channel is translated independently.
- Pixels that move outside the image boundaries are reflected back
  to avoid border artifacts.
- Uses `@inbounds` and `@inline` for performance.
- Raises an error if translation exceeds half the image size.
"""
function translate!(out::Array{T,3}, img::Array{T,3}, dx::Int, dy::Int) where T
    H, W, C = size(img)
    @assert size(out) == (H, W, C) "Output buffer must have the same size as input image."
    
    # Check that translation does not exceed half the image size
    @assert abs(dx) <= H ÷ 2 "dx translation exceeds half the image height."
    @assert abs(dy) <= W ÷ 2 "dy translation exceeds half the image width."

    # Helper function: reflect indices for out-of-bounds positions
    @inline reflect(i, n) = ifelse(i < 1, 2 - i, ifelse(i > n, 2n - i, i))

    # Iterate over all pixels and channels
    @inbounds for ci in 1:C, yi in 1:W, xi in 1:H
        sx = reflect(xi - dx, H)  # source x-coordinate after reflection
        sy = reflect(yi - dy, W)  # source y-coordinate after reflection
        out[xi, yi, ci] = img[sx, sy, ci]  # copy pixel from source
    end

    return out
end