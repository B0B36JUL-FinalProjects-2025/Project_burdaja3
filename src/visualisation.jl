using UMAP
using Plots

include("metric.jl")

function visualise(data::Array{Float32, 2}, labels::Vector{Int})
    distan = distances(data)

    Y = umap(
        distan,
        2;
        metric=:precomputed
    )

    

    Y = Y'

    sc = scatter(
        Y[:, 1],
        Y[:, 2],
        group = labels,
        legend = :outertopright,
        title = "Embeddings"
    )
    display(sc)

    return Y
end