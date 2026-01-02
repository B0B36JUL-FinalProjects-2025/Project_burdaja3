using UMAP
using Plots

include("metric.jl")

function visualise2D(data::Array{Float32, 2}, labels::Vector{Int})
    distan = distances(data)

    Y = umap(
        distan,
        2;
        metric=:precomputed
    )


    sc = scatter(
        Y[1, :],
        Y[2, :],
        group = labels,
        legend = :outertopright,
        title = "Embeddings"
    )
    display(sc)

    return Y
end


function visualise3D(data::Array{Float32, 2}, labels::Vector{Int})
    distan = distances(data)

    Y = umap(
        distan,
        3;             
        metric = :precomputed
    )

    sc = scatter3d(
        Y[1, :],
        Y[2, :],
        Y[3, :],
        group = labels,
        legend = :outertopright,
        title = "Embeddings"
    )

    display(sc)

    return Y
end