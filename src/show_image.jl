using Images, ImageView

"""
    show_image(img::Array{UInt8,3})

Displays a 3D image (H × W × C) represented by an array of `UInt8` type as an RGB image.

# Arguments
- `img::Array{UInt8,3}` : The input image (H × W × C) with `UInt8` pixel values.
"""
function show_image(img::Array{UInt8,3})
    img = permutedims(img, (3, 1, 2))
    img_f = float32.(img)         
    img_rgb = colorview(RGB, img_f)
    imshow(img_rgb)
end



"""
    show_image(img::Array{Float32,3})

Displays a 3D image (H × W × C) represented by an array of `Float32` type as an RGB image.

# Arguments
- `img::Array{Float32,3}` : The input image (H × W × C) with `Float32` pixel values.
"""
function show_image(img::Array{Float32,3})
    img_f = permutedims(img, (3, 1, 2))       
    img_rgb = colorview(RGB, img_f)
    imshow(img_rgb)
end


"""
    show_image(images::AbstractArray{T,4}, index::Int) where T<:Union{UInt8,Float32}

Displays a single image from a 4D array of images by extracting it using the specified index.

# Arguments
- `images::AbstractArray{T,4}` : A 4D array of images (e.g., a batch of images) with `T` type (`UInt8` or `Float32`).
- `index::Int` : The index of the image to display.
"""
function show_image(images::AbstractArray{T,4}, index::Int) where T<:Union{UInt8,Float32}
    img = images[:, :, :, index]
    show_image(img)
end