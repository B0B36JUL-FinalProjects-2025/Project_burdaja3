using Flux
using Flux: onehotbatch, crossentropy
using Statistics: mean
using Random

include("load_galaxy.jl")
include("augmentation/augment.jl")
include("show_image.jl")

function get_batch(images::Array{UInt8,4}, labels::Vector{UInt8}, batch_size::Int)
    N = size(images,4)
    idx = rand(1:N, batch_size)
    x = Float32.(images[:, :, :, idx]) ./ 255f0

    augment!(x)
    
    y = onehotbatch(labels[idx], 0:9) 

    return x, y
end


function train_test_split(images::Array{UInt8,4}, labels::Vector{UInt8}, train_frac=0.8)
    N = size(images,4)
    idx = shuffle(1:N)
    n_train = Int(floor(train_frac * N))
    train_idx = idx[1:n_train]
    test_idx = idx[n_train+1:end]
    
    train_images = images[:, :, :, train_idx]
    train_labels = labels[train_idx]
    
    test_images = images[:, :, :, test_idx]
    test_labels = labels[test_idx]
    
    return (train_images, train_labels), (test_images, test_labels)
end

function accuracy(pred, y)
    y_true = Flux.onecold(y, 0:9)     
    y_pred = Flux.onecold(pred, 0:9)    
    return mean(y_pred .== y_true)      
end

function train_model()
    images, labels = load_galaxy("data/Galaxy10_DECals.h5") 
    println("Dataset loaded")

    (train_images, train_labels), (test_images, test_labels) = train_test_split(images, labels)

    model = Chain(
        # 256×256×3
        Conv((3,3), 3=>32, relu, pad=1),
        MaxPool((2,2)),          # 128×128×32

        Conv((3,3), 32=>64, relu, pad=1),
        MaxPool((2,2)),          # 64×64×64

        Conv((3,3), 64=>128, relu, pad=1),
        MaxPool((2,2)),          # 32×32×128

        Conv((3,3), 128=>256, relu, pad=1),
        MaxPool((2,2)),          # 16×16×256

        Conv((3,3), 256=>512, relu, pad=1),
        MaxPool((2,2)),          # 8×8×512

        x -> mean(x, dims=(1,2)),  # 1×1×512×B
        Flux.flatten,              # 512×B

        Dense(512, 10),
        Flux.softmax
    )

    opt = ADAM(0.005)
    opt_state = Flux.setup(opt, model)

    loss_fn(m, x, y) = crossentropy(m(x), y)

    

    for batch in 1:10000
        batch_size = 32
        xb, yb = get_batch(train_images, train_labels, batch_size)
       
        gs = Flux.gradient(m -> loss_fn(m, xb, yb), model)

        Flux.update!(opt_state, model, gs[1])
        
        if batch % 10 == 0
            @info "Batch $batch" loss=loss_fn(model, xb, yb)
        else
            @info "Batch $batch"
        end
    end
end

