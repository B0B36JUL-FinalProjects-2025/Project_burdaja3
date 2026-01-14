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
    loss_fn(model, x, y, metric_losses::Vector{MetricLoss})

Compute training loss combining classification and metric loss.

# Arguments
- `model`: hybrid model
- `x`: input batch
- `y`: target labels (one-hot encoded)
- `metric_losses::Vector{MetricLoss}`: metric losses to be used

# Returns
- loss value
"""
function loss_fn(model, 
                x, 
                y, 
                metric_losses::Vector{MetricLoss}=MetricLoss[
                    ContrastiveLoss(0.7f0, 1.6f0),
                    TripletLoss(0.4f0, 0.8f0)
                ])
    logits = model(x)
    loss = logitcrossentropy(logits, y)

    embedding = get_embs(model, x)
    labels = Flux.onecold(y, 0:9)

    D = distances(embedding) 

    for metric_loss in metric_losses
        loss += metric_loss_fn(metric_loss, D, labels)
    end
        
    return loss
end


"""
    loss_fn(model, x, y)

Compute training loss.

# Arguments
- `model`: hybrid model
- `x`: input batch
- `y`: target labels (one-hot encoded)

# Returns
- loss value
"""
function loss_fn(model, x, y)
    logits = model(x)
    loss = logitcrossentropy(logits, y)
    return loss
end



"""
    train_model(;
            augment::Bool=true, 
            batches::Int=2000, 
            block_size::Int=16,  
            save_path::String="model/cnn_metric.bson", 
            resume::Bool=true, 
            use_metric_learning::Bool=true,
            metric_losses::Vector{MetricLoss})

Train the hybrid model on the training dataset.

# Keyword Arguments
- `augment::Bool`: apply data augmentation during training
- `batches::Int`: number of training iterations
- `block_size::Int`: number of same labeled images in one batch, batch size = block size * number of distinct classes
- `save_path::String`: path to save and load the model
- `resume::Bool`: resume training from saved checkpoint if available
- `use_metric_learning::Bool`: use metric loss in loss
- `metric_losses::Vector{MetricLoss}`: metric losses to be used
"""
function train_model(;
            augment::Bool=true, 
            batches::Int=2000, 
            block_size::Int=16,  
            save_path::String="model/cnn_metric.bson", 
            resume::Bool=true, 
            use_metric_learning::Bool=true,
            metric_losses::Vector{MetricLoss}=MetricLoss[
                ContrastiveLoss(0.7f0, 1.6f0),
                TripletLoss(0.4f0, 0.8f0)
            ])


    model = get_default_hybrid_model()
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

        if use_metric_learning
            gs = Flux.gradient(m -> loss_fn(m, xb, yb, metric_losses), model)
        else
            gs = Flux.gradient(m -> loss_fn(m, xb, yb), model)
        end
        

        Flux.update!(opt_state, model, gs[1])
        
        if batch % 10 == 0
            println(batch)
            if use_metric_learning
                println(loss_fn(model, xb, yb, metric_losses))
            else
                println(loss_fn(model, xb, yb))
            end
        end
    end

    save_model(save_path, model, opt_state, batches + 1)
end

