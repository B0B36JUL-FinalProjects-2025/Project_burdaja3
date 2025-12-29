using Flux
using Flux: onehotbatch, logitcrossentropy
using Statistics: mean
using Random
using BSON

include("dataset_loading.jl")
include("batches.jl")
include("augmentation/augment.jl")
include("metric.jl")

function accuracy(pred, y)
    y_true = Flux.onecold(y, 0:9)     
    y_pred = Flux.onecold(pred, 0:9)    
    return mean(y_pred .== y_true)      
end

function l2_normalize(x; dims=1)
    return x ./ sqrt.(sum(x.^2, dims=dims) .+ 1e-8)
end

struct HybridModel
    embedding
    classifier
end

(m::HybridModel)(x) = m.classifier(m.embedding(x))

get_embs(m::HybridModel, x) = m.embedding(x)

get_probs(m::HybridModel, x) = softmax(m(x), dims=1)

get_outputs(m::HybridModel, x) = (m.embedding(x), m.classifier(m.embedding(x)))


function loss_fn(model, x, y, alpha)

    embedding_raw, logits = get_outputs(model, x)
    embedding_norm = l2_normalize(embedding_raw, dims=1)

    labels = Flux.onecold(y, 0:9)
    me_loss = metric_loss(embedding_norm, labels)
    ce_loss = logitcrossentropy(logits, y)

    return ce_loss  + alpha * me_loss
end



function train_model(;augment::Bool=true, batches::Int=10000, block_size::Int=32, alpha::Float32=1f0, save_path::String="model/cnn_metric.bson")
    embedding = Chain(
        Conv((3,3), 3=>8, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 8=>16, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 16=>32, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 32=>64, relu, pad=1),
        x -> mean(x, dims=(1,2)),

        Flux.flatten
    )

    classifier = Dense(64, 10)

    model = HybridModel(embedding, classifier)


    opt = ADAM(0.005)
    opt_state = Flux.setup(opt, model)

    

    train = load_train()

    for batch in 1:batches

        xb, yb = load_batch(train, block_size)

        # augmentations can also be added directly to the datasets via dataset_enlargement.jl
        if(augment)
            augment!(xb,xb)
        end
       
        gs = Flux.gradient(m -> loss_fn(m, xb, yb, alpha), model)

        Flux.update!(opt_state, model, gs[1])
        
        if batch % 10 == 0
            println(batch)
            println(loss_fn(model, xb, yb, alpha))
            println(accuracy(model(xb),yb))
            if batch % 100 == 0
                test_x, test_y = load_test()
                pred = model((Float32.(test_x)) ./ 255f0)
                y = onehotbatch(test_y, 0:9)
                println(accuracy(pred, y))
            end
        else
            println(batch)
        end
    end

    save_model(save_path, model)
end

function save_model(path, model)
    BSON.@save path state=Flux.state(model)
end

function load_model(path, model)
    BSON.@load path state
    Flux.loadmodel!(model, state)
    return model
end

