using ImageTransformations, Interpolations, Images

include("rotation.jl")
include("translation.jl")
include("zoom.jl")
include("noise.jl")

"""
    augment!(out, imgs;
             max_rotate=180f0,
             max_transpose=0.25f0,
             max_zoom=1.25f0
             stddev_perc=0.05
             )

Applies random augmentation to a batch of images.

Arguments
- `out::Array{Float32,4}` :
    Output buffer (H × W × C × N)

- `imgs::Array{Float32,4}` :
    Input batch (H × W × C × N)

Keyword arguments
- `max_rotate`    : maximum rotation angle in degrees
- `max_transpose` : maximum translation as fraction of image height
- `max_zoom`      : maximum zoom factor (≥ 1)
- `stddev_perc`   : standard deviation of the Gaussian noise in percentage
- `max_value`     : the maximum value for pixel intensities (default is 1.0, suitable for images normalized between 0 and 1).

Behavior
For each image:
1. rotate
2. translate
3. zoom
4. add noise
"""
function augment!(
    out::Array{Float32,4},
    imgs::Array{Float32,4};
    max_rotate::Float32 = 180f0,
    max_transpose::Float32 = 0.25f0,
    max_zoom::Float32 = 1.25f0,
    stddev_perc::Float32 = 0.05f0,
    max_value::Float32 = 1.0f0
)
    H, W, C, N = size(imgs)
    @assert size(out) == (H, W, C, N)

    max_trans_h = floor(Int, H * max_transpose)
    max_trans_w = floor(Int, W * max_transpose)

    stddev = stddev_perc*max_value

    # working buffers (reused for all images)
    buf1 = Array{Float32}(undef, H, W, C)
    buf2 = Array{Float32}(undef, H, W, C)
    buf3 = Array{Float32}(undef, H, W, C)

    for n in 1:N
        img  = @view imgs[:,:,:,n]
        outn = @view out[:,:,:,n]

        θ  = (2f0 * rand(Float32) - 1f0) * max_rotate
        dx = rand(-max_trans_h:max_trans_h)
        dy = rand(-max_trans_w:max_trans_w)
        z  = 1f0 + rand(Float32) * (max_zoom - 1f0)

        rotate!(buf1, img, θ)
        translate!(buf2, buf1, dx, dy)    
        zoom!(buf3, buf2, z)
        noise!(outn, buf3, stddev = stddev)
    end

    out .= clamp.(out, 0.0f0, max_value)

    return out
end




"""
    augment!(out, img; ...)

Applies random augmentation to a single image.

Arguments:
- `out::Array{Float32,3}` : output buffer
- `img::Array{Float32,3}` : input image

Keyword arguments:
- `max_rotate`    : max rotation angle in degrees
- `max_transpose` : max translation as fraction of image height
- `max_zoom`      : maximum zoom factor
- `stddev_perc`   : standard deviation of the Gaussian noise in percentage
- `max_value`     : the maximum value for pixel intensities (default is 1.0, suitable for images normalized between 0 and 1).

Behavior
For each image:
1. rotate
2. translate
3. zoom
4. add noise
"""
function augment!(
    out::Array{Float32,3},
    img::Array{Float32,3};
    max_rotate::Float32 = 180f0,
    max_transpose::Float32 = 0.25f0,
    max_zoom::Float32 = 1.25f0,
    stddev_perc::Float32 = 0.05f0,
    max_value::Float32 = 1.0f0
)
    H, W, C = size(img)
    @assert size(out) == (H, W, C)

    max_trans_h = floor(Int, H * max_transpose)
    max_trans_w = floor(Int, W * max_transpose)

    stddev = stddev_perc*max_value

    buf1 = Array{Float32}(undef, H, W, C)
    buf2 = Array{Float32}(undef, H, W, C)
    buf3 = Array{Float32}(undef, H, W, C)

    θ  = (2f0 * rand(Float32) - 1f0) * max_rotate
    dx = rand(-max_trans_h:max_trans_h)
    dy = rand(-max_trans_w:max_trans_w)
    z  = 1f0 + rand(Float32) * (max_zoom - 1f0)

    rotate!(buf1, img, θ)
    translate!(buf2, buf1, dx, dy)
    zoom!(buf3, buf2, z)
    noise!(out, buf3, stddev = stddev)

    out .= clamp.(out, 0.0f0, max_value)

    return out
end



