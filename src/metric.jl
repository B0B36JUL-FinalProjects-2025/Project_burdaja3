"""
    l2_normalize(x; dims=1)

L2-normalize tensor along a given dimension.

A small epsilon is used to avoid division by zero and ensure
numerical stability during training.

# Arguments
- `x`: input array (e.g. embedding matrix of shape (D, B))
- `dims`: dimension along which normalization is applied (default: 1)

# Returns
- Array of the same shape as `x`, where each vector along `dims`
  has unit L2 norm.
"""
function l2_normalize(x; dims=1)
    norms = sqrt.(sum(x.^2, dims=dims))
    norms = max.(norms, 1e-6)
    return x ./ norms
end



"""
    distances(emb::Array{Float32,2})

Compute pairwise L2 distances between all embeddings in the batch.

# Arguments
- `emb::Array{Float32,2}`: embeddings (embedding_dim, batch_size)

# Returns
- `D::Array`: B×B matrix of pairwise Euclidean distances
"""
function distances(emb::Array{Float32,2})
    emb = l2_normalize(emb)

    sq = sum(emb.^2, dims=1)               
    dist2 = sq .+ sq' .- 2.0 .* (emb' * emb)
    dist2 = max.(dist2, 1e-12)
    return sqrt.(dist2)
end

















abstract type MetricLoss end

struct TripletLoss <: MetricLoss
    alpha::Float32
    margin::Float32
end

struct ContrastiveLoss <: MetricLoss
    alpha::Float32
    margin::Float32
end


"""
    metric_loss_fn(loss_type::ContrastiveLoss, D::Matrix{Float64}, labels::Vector{Int})

Compute contrastive loss on a batch of embeddings.

# Arguments
- `loss_type::TripletLoss`: triplet loss struct for multiple dispatch
- `D::Matrix{Float64}`: embeddings distance matrix
- `labels::Vector{Int}`: labels

# Returns
- `loss::Float32`: scalar contrastive loss

Positive pairs are pulled together; negative pairs are pushed apart beyond the margin.
"""
function metric_loss_fn(loss_type::ContrastiveLoss, D::Matrix{Float64}, labels::Vector{Int})
    pos_mask = labels .== labels' # matrix of positive doubles
    neg_mask = .!pos_mask # matrix of negative embedings doubles

    pos_loss = D[pos_mask] .^ 2 # minimizing distances of same-labeled embeddings
    neg_loss = max.(0f0, loss_type.margin .- D[neg_mask]) .^ 2 # maximizing distances of differently labeled embeddings

    sum_pos_loss = sum(pos_loss) / length(pos_loss)
    sum_neg_loss = sum(neg_loss) / length(neg_loss)

    return loss_type.alpha * (sum_pos_loss + sum_neg_loss)

end


"""
    triplet_loss(loss_type::TripletLoss, D::Matrix{Float64}, labels::Vector{Int})
    
Compute batch-hard triplet margin loss on a batch.

# Arguments
- `loss_type::TripletLoss`: triplet loss struct for multiple dispatch
- `D::Matrix{Float64}`: embeddings distance matrix
- `labels::Vector{Int}`: labels

# Returns
- `loss::Float32`: scalar triplet loss

For each anchor, selects hardest positive (furthest in same class) and hardest negative
(closest in different class) and applies margin ranking loss.
"""
function metric_loss_fn(loss_type::TripletLoss, D::Matrix{Float64}, labels::Vector{Int})
    B = length(labels)

    loss = 0f0
    count = 0

    for i in 1:B
        same = Array(labels .== labels[i])   # mask for positives (same class) 
        diff = Array(labels .!= labels[i])   # mask for negatives (different class) 

        if !any(same) || !any(diff)
            continue                
        end

        hardest_pos = maximum(D[i, same])   # furthest positive in batch
        hardest_neg = minimum(D[i, diff])   # closest negative in batch

        l = max(0f0, hardest_pos - hardest_neg + loss_type.margin)  # batch-hard triplet loss
        loss += l
        count += 1
    end

    tri_loss = count > 0 ? loss / count : 0f0  # average over valid anchors

    return loss_type.alpha * tri_loss

end

