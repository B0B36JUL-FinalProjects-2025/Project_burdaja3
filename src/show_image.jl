using Images, ImageView

function show_image(img::Array{UInt8,3})
    img = permutedims(img, (3, 1, 2))
    img_f = float32.(img) ./ 255         
    img_rgb = colorview(RGB, img_f)
    imshow(img_rgb)
end

function show_image(img::Array{Float32,3})
    img_f = permutedims(img, (3, 1, 2))       
    img_rgb = colorview(RGB, img_f)
    imshow(img_rgb)
end

function show_image(images::AbstractArray{T,4}, index::Int) where T<:Union{UInt8,Float32}
    img = images[:, :, :, index]
    show_image(img)
end