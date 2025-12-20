using Flux
using Flux: onehotbatch, crossentropy
using Statistics: mean
using Random

include("load_galaxy.jl")

function get_batch(images::Array{UInt8,4}, labels::Vector{UInt8}, batch_size::Int)
    N = size(images,4)
    idx = rand(1:N, batch_size)
    x = Float32.(images[:, :, :,idx]) ./ 255.0

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

    println(size(images))

    (train_images, train_labels), (test_images, test_labels) = train_test_split(images, labels)

    model = Chain(
        Conv((3,3), 3=>8, relu),
        MaxPool((2,2)),
        Conv((3,3), 8=>16, relu),
        MaxPool((2,2)),
        Conv((3,3), 16=>32, relu),
        MaxPool((2,2)),
        Conv((3,3), 32=>64, relu),
        MaxPool((2,2)),
        Conv((3,3), 64=>128, relu),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(4608, 128, relu),
        Dense(128, 10),
        Flux.softmax
    )

    opt = ADAM(0.0005)
    opt_state = Flux.setup(opt, model)

    loss_fn(m, x, y) = crossentropy(m(x), y)

    

    for batch in 1:10000
        batch_size = 256
        xb, yb = get_batch(train_images, train_labels, batch_size)
       
        gs = Flux.gradient(m -> loss_fn(m, xb, yb), model)

        Flux.update!(opt_state, model, gs[1])
        
        if batch % 10 == 0
            tx, ty = get_batch(test_images, test_labels, batch_size)
            train_acc = accuracy(model(xb), yb)
            test_acc = accuracy(model(tx), ty)
            @info "Batch $batch" loss=loss_fn(model, xb, yb) test_loss=loss_fn(model, tx, ty) train_acc=train_acc test_acc=test_acc
        end
    end
end