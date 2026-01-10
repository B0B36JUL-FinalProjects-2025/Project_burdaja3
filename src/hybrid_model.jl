using Flux
using Flux: onehotbatch, onecold
using Statistics

"""
    HybridModel

Simple hybrid model combining an embedding network and a classifier.

# Fields
- `embedding`: neural network producing feature embeddings
- `classifier`: classification head mapping embeddings to class logits
"""
struct HybridModel
    embedding
    classifier
end

"""
    get_defualt_hybrid_model()

Creates and returns the default HybridModel consisting of an embedding module and a classifier module.

# Returns
- `HybridModel` : A model composed of the embedding layer and classifier.
"""
function get_defualt_hybrid_model()
    embedding = Chain(
        Conv((3,3), 3=>8, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 8=>16, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 16=>32, relu, pad=1),
        MaxPool((2,2)),

        Conv((3,3), 32=>64, relu, pad=1),
        MaxPool((2,2)),

        x -> mean(x, dims=(1,2)),

        Flux.flatten,

        Dense(64, 256, relu),
    )

    classifier = Dense(256, 10)

    model = HybridModel(embedding, classifier)

    return model
end





"""
    (m::HybridModel)(x) = m.classifier(m.embedding(x))

Performs a prediction using the HybridModel on the input data `x`.

# Arguments
- `m::HybridModel` : The model containing both embedding and classifier components.
- `x::AbstractArray` : The input data for which predictions are made.

# Returns
- `Array` : The output from the classifier module of the model based on the input `x`.
"""
(m::HybridModel)(x) = m.classifier(m.embedding(x))


"""
    get_embs(m::HybridModel, x) = m.embedding(x)

Extracts the embedding representations for the input data `x` using the embedding part of the HybridModel.

# Arguments
- `m::HybridModel` : The model containing the embedding layer.
- `x::AbstractArray` : The input data for which embedding representations are obtained.

# Returns
- `Array` : The embedded representations of the input data `x` from the model's embedding module.
"""
get_embs(m::HybridModel, x) = m.embedding(x)



"""
    get_probs(m::HybridModel, x) = softmax(m(x), dims=1)

Computes the probabilities for each class using the softmax function on the model's outputs.

# Arguments
- `m::HybridModel` : The model containing both embedding and classifier components.
- `x::AbstractArray` : The input data for which class probabilities are computed.

# Returns
- `Array` : The computed probabilities for each class.
"""
get_probs(m::HybridModel, x) = softmax(m(x), dims=1)





"""
    get_class(m::HybridModel, x) = onecold(m(x), 0:9)

Determines the class with the highest probability for the input data `x` using the HybridModel.

# Arguments
- `m::HybridModel` : The model containing both embedding and classifier components.
- `x::AbstractArray` : The input data for which the predicted class is determined.

# Returns
- `Int` : The class with the highest probability (within the range 0:9).
"""
get_class(m::HybridModel, x) = onecold(m(x), 0:9)


