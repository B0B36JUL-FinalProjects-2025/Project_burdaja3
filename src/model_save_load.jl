using BSON
using Flux

"""
    save_model(path, model, opt_state, step)

Saves the model state, optimizer state, and batch to the specified file path.

# Arguments
- `path::String` : The file path where the model state will be saved.
- `model` : The model whose state is to be saved.
- `opt_state` : The state of the optimizer.
- `step::Int` : The current training batch.
"""
function save_model(path, model, opt_state, step)
    BSON.@save path state=Flux.state(model) opt_state step
end


"""
Loads the model state, optimizer state, and batch from the specified file path.

# Arguments
- `path::String` : The file path where the model state is stored.
- `model` : The model to load the state into.

# Returns
- `opt_state` : The loaded optimizer state.
- `step::Int` : The loaded training batch.
"""
function load_model!(path, model)
    BSON.@load path state opt_state step
    Flux.loadmodel!(model, state)
    return opt_state, step
end