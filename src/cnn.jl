using Flux
using Flux: onehotbatch, logitcrossentropy
using Random
using BSON

include("dataset_loading.jl")
include("batches.jl")
include("augmentation/augment.jl")
include("metric.jl")
include("hybrid_model.jl")
include("visualisation.jl")
include("model_save_load.jl")

"""
    loss_fn(model, x, y, alpha)

Compute training loss combining classification and metric loss.

# Arguments
- `model`: hybrid model
- `x`: input batch
- `y`: target labels (one-hot encoded)
- `alpha`: weight of the metric loss term

# Returns
- loss value
"""
function loss_fn(model, x, y, alpha)
    embedding = get_embs(model, x)
    logits = model(x)

    ce_loss = logitcrossentropy(logits, y)

    if (alpha > 0)
        lab = Flux.onecold(y, 0:9)
        me_loss = metric_loss(Float32.(embedding), lab)
        
        return ce_loss + alpha * me_loss
    end

    return ce_loss
end


"""
    train_model(; augment=true, batches=2000, block_size=16,
                  alpha=0.7f0, save_path="model/cnn_metric.bson",
                  resume=true)

Train the hybrid model on the training dataset.

# Keyword Arguments
- `augment::Bool`: apply data augmentation during training
- `batches::Int`: number of training iterations
- `block_size::Int`: number of same labeled images in one batch, batch size = block size * number of distinct classes
- `alpha::Float32`: weight of metric loss
- `save_path::String`: path to save and load the model
- `resume::Bool`: resume training from saved checkpoint if available
"""
function train_model(;augment::Bool=true, batches::Int=2000, block_size::Int=16, alpha::Float32=0.7f0, save_path::String="model/cnn_metric.bson", resume::Bool=true)
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
        end
    end

    save_model(save_path, model, opt_state, batches + 1)
end

