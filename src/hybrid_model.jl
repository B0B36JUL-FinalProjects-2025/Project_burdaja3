struct HybridModel
    embedding
    classifier
end

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

(m::HybridModel)(x) = m.classifier(m.embedding(x))

get_embs(m::HybridModel, x) = m.embedding(x)

get_probs(m::HybridModel, x) = softmax(m(x), dims=1)

get_class(m::HybridModel, x) = onecold(m(x), 0:9)


