using Images, ImageView

function show_image(img::Array{UInt8,3})
    size(img)
    img_f = float32.(img) ./ 255         
    img_rgb = colorview(RGB, img_f)
    imshow(img_rgb)
end

function show_image(images::Array{UInt8,4}, index::Int)
    img = images[index,:, :, :]
    show_image(img)
end