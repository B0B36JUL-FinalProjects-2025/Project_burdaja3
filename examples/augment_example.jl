include("../src/dataset_loading.jl")
include("../src/augmentation/augment.jl")
include("../src/show_image.jl")


function augment_example(path::String = "data/test")
    imgs, labels = load_test(path)

    idx = rand(1:size(imgs, 4))

    img = float32.(imgs[:,:,:,idx]) / 255

    augmented = similar(img)

    augment!(augmented, img)

    show_image(img)
    show_image(augmented)
end