using ImageTransformations, Interpolations, Images

"""
    zoom!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, scale::Float32)

Performs a zoom-in operation on a 3D Float32 image (H × W × C) and writes the result
into the pre-allocated `out` buffer. Only supports `scale >= 1`.

# Arguments
- `out::AbstractArray{Float32,3}` : pre-allocated output buffer of the same size as `img`.
- `img::AbstractArray{Float32,3}` : input image to zoom.
- `scale::Float32` : zoom factor. Must be >= 1.0.  

# Returns
- `out` : the zoomed image (mutated in-place).

# Behavior
- If `scale == 1`, the input image is copied directly into `out`.
- If `scale < 1`, throws an error.
- Zooming is done by resizing each channel individually and cropping
  the center H×W region to match the original image size.
- No extra buffers are allocated except for the temporary output of `imresize`.
"""
function zoom!(out::AbstractArray{Float32,3}, img::AbstractArray{Float32,3}, scale::Float32)
    H, W, C = size(img)
    @assert size(out) == (H, W, C) "Output buffer must have the same size as the input."

    # Handle edge cases
    if scale < 1f0
        error("zoom! supports only scale >= 1")
    elseif scale == 1f0
        # No zoom, just copy input to output
        out .= img
        return out
    end

    # Process each channel independently
    for c in 1:C
        # Resize current channel
        resized = imresize(img[:,:,c], ratio=scale, method=BSpline(Linear()))
        Hc, Wc = size(resized)

        # Compute starting indices to crop the center H×W region
        startx = Int(floor((Hc - H)/2)) + 1
        starty = Int(floor((Wc - W)/2)) + 1

        # Copy cropped center region into output buffer
        out[:,:,c] .= @view resized[startx:startx+H-1, starty:starty+W-1]
    end

    return out
end
