using ImageTransformations, Interpolations, Images

include("translation.jl")
include("rotation.jl")
include("zoom.jl")

"""
    augment!(imgs::Array{Float32,4};
             max_rotate::Float32 = 180f0,
             max_transpose::Float32 = 0.1f0,
             max_zoom::Float32 = 1.1f0)

Applies in-place random augmentation to a batch of 3D Float32 images (H × W × C × N).
Supports rotation, translation, and zooming, using pre-allocated buffers to avoid
allocations inside the loop.

# Arguments
- `imgs::Array{Float32,4}` : batch of images to augment (H × W × C × N).
- `max_rotate::Float32` : maximum rotation angle in degrees (±max_rotate).
- `max_transpose::Float32` : maximum fraction of image height for translation.
- `max_zoom::Float32` : maximum zoom factor (scale between 1 and max_zoom).

# Behavior
- Iterates over each image in the batch.
- Generates random augmentation parameters for rotation, translation, and zoom.
- Uses `rotate!`, `translate!`, and `zoom!` mutating functions.
- Pre-allocates two working buffers (`buf1`, `buf2`) for efficiency.
- Applies operations in the order: rotate → translate → zoom.
- Mutates `imgs` in-place.
"""
function augment!(
    imgs::Array{Float32,4};
    max_rotate::Float32 = 180f0,
    max_transpose::Float32 = 0.1f0,
    max_zoom::Float32 = 1.1f0,
)

    H, W, C, N = size(imgs)
    max_trans = floor(Int, H * max_transpose)  # max pixel shift for translation

    # Pre-allocated working buffers to avoid allocations in the loop
    buf1 = Array{Float32}(undef, H, W, C)
    buf2 = Array{Float32}(undef, H, W, C)

    for n in 1:N
        # Work directly on a view of the nth image
        img = @view imgs[:,:,:,n]

        # Random augmentation parameters
        θ  = (2f0 * rand(Float32) - 1f0) * max_rotate  # rotation angle in [-max_rotate, max_rotate]
        dx = rand(-max_trans:max_trans)                 # x-axis translation
        dy = rand(-max_trans:max_trans)                 # y-axis translation
        z  = 1f0 + rand(Float32) * (max_zoom - 1f0)    # zoom factor >= 1

        # Apply augmentations using mutating functions
        rotate!(buf1, img, θ)          # rotate img → buf1
        translate!(buf2, buf1, dx, dy) # translate buf1 → buf2
        zoom!(img, buf2, z)            # zoom buf2 → img (in-place)

    end

    return nothing
end

