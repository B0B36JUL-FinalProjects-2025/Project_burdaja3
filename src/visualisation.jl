using UMAP
using Plots

include("metric.jl")

"""
    visualise2D(data::Array{Float32, 2}, labels::Vector{Int})

Visualizes embeddings by applying UMAP dimensionality reduction and creating a scatter plot.

# Arguments
- `data::Array{Float32,2}` : The matrix of embeddings
- `labels::Vector{Int}` : A vector of labels corresponding to each sample in the dataset.

# Returns
- `Array{Float32,2}` : The UMAP embeddings for the input data, returned as a 2 × N array.
"""
function visualise2D(data::Array{Float32, 2}, labels::Vector{Int})
    distan = distances(data)

    Y = UMAP.umap(
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

"""
    visualise3D(data::Array{Float32, 2}, labels::Vector{Int})

Visualizes embeddings by applying UMAP dimensionality reduction and creating a 3D scatter plot.

# Arguments
- `data::Array{Float32,2}` : The matrix of embeddings
- `labels::Vector{Int}` : A vector of labels corresponding to each sample in the dataset.

# Returns
- `Array{Float32,2}` : The UMAP embeddings for the input data, returned as a 3 × N array.
"""
function visualise3D(data::Array{Float32, 2}, labels::Vector{Int})
    distan = distances(data)

    Y = UMAP.umap(
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