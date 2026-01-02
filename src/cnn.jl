using Flux
using Flux: onehotbatch, logitcrossentropy
using Statistics: mean
using Random
using BSON

include("dataset_loading.jl")
include("batches.jl")
include("augmentation/augment.jl")
include("metric.jl")
include("hybrid_model.jl")
include("visualisation.jl")

function accuracy(pred, y)
    y_true = Flux.onecold(y, 0:9)     
    y_pred = Flux.onecold(pred, 0:9)    
    return mean(y_pred .== y_true)      
end

function loss_fn(model, x, y, alpha)
    embedding_raw = get_embs(model, x)
    logits = model(x)

    ce_loss = logitcrossentropy(logits, y)

    if (alpha > 0)
        embedding_norm = l2_normalize(embedding_raw, dims=1)
        lab = Flux.onecold(y, 0:9)
        me_loss = metric_loss(Float32.(embedding_norm), lab)
        
        return ce_loss + alpha * me_loss
    end

    return ce_loss
end



function train_model(;augment::Bool=true, batches::Int=2000, block_size::Int=16, alpha::Float32=0.7f0, save_path::String="model/cnn_metric.bson", resume::Bool=false)
    model = get_defualt_hybrid_model()
    if resume && isfile(save_path)
        opt_state, start_batch = load_model!(save_path, model)
    else
        start_batch = 1
        opt = ADAM()
        opt_state = Flux.setup(opt, model)
    end

    train = load_train()

    for batch in start_batch:batches

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

                visualise(get_embs(model, test_x), test_y)
            end
        end
    end

    save_model(save_path, model, opt_state, batches + 1)
end

function save_model(path, model, opt_state, step)
    BSON.@save path state=Flux.state(model) opt_state step
end

function load_model!(path, model)
    BSON.@load path state opt_state step
    Flux.loadmodel!(model, state)
    return opt_state, step
end