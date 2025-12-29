"""
Compute combined metric loss (triplet + contrastive) on a batch of embeddings.

# Arguments
- `emb::Array`: embeddingy batchu, shape (embedding_dim, batch_size)
- `y`: labels
- `use_triplet::Bool`: whether to include triplet loss (default: true)
- `use_contrastive::Bool`: whether to include contrastive loss (default: true)
- `margin_triplet::Float32`: margin for triplet loss (default: 0.5)
- `margin_contrastive::Float32`: margin for contrastive loss (default: 1.0)

# Returns
- `loss::Float32`: scalar loss combining requested metric losses
"""
function metric_loss(emb::Array{Float64,2}, y::Vector{Int};
        use_triplet::Bool = true,
        use_contrastive::Bool = true,
        margin_triplet::Float32 = 0.5f0,
        margin_contrastive::Float32 = 1.0f0
    )

    loss = 0f0

    if use_triplet
        loss += triplet_loss(emb, y; margin=margin_triplet)
    end

    if use_contrastive
        loss += contrastive_loss(emb, y; margin=margin_contrastive)
    end

    return loss
end


"""
Compute pairwise L2 distances between all embeddings in the batch.

# Arguments
- `emb::Array`: embeddingy batchu, shape (embedding_dim, batch_size)

# Returns
- `D::Array`: B×B matrix of pairwise Euclidean distances
"""
function distances(emb::Array{Float64,2})
    # ∣∣ei​−ej​∣∣2 = ∣∣ei​∣∣2 + ∣∣ej​∣∣2 − 2(eiT​ej​)
    sq = sum(emb .^ 2, dims=1)               
    dist2 = sq .+ sq' .- 2f0 .* (emb' * emb)
    return sqrt.(max.(dist2, 0f0))
end


"""
Compute contrastive loss on a batch of embeddings.

# Arguments
- `emb::Array`: embeddings, shape (embedding_dim, batch_size)
- `labels`: labels
- `margin::Float32`: distance margin for negative pairs (default: 1.0)

# Returns
- `loss::Float32`: scalar contrastive loss

Positive pairs are pulled together; negative pairs are pushed apart beyond the margin.
"""
function contrastive_loss(emb::Array{Float64,2}, labels::Vector{Int}; margin::Float32=1.0f0)
    D = distances(emb)       

    pos_mask = labels .== labels' # matrix of positive doubles
    neg_mask = .!pos_mask # matrix of negative embedings doubles

    pos_loss = D[pos_mask] .^ 2 # minimizing distances of same-labeled embeddings
    neg_loss = max.(0f0, margin .- D[neg_mask]) .^ 2 # maximizing distances of differently labeled embeddings

    return (sum(pos_loss) + sum(neg_loss)) / (length(pos_loss) + length(neg_loss)) # normalize by number of pairs
end


"""
Compute batch-hard triplet margin loss on a batch of embeddings.

# Arguments
- `emb::Array`: embeddings, shape (embedding_dim, batch_size)
- `labels`: labels
- `margin::Float32`: margin for triplet loss (default: 0.5)

# Returns
- `loss::Float32`: scalar triplet loss

For each anchor, selects hardest positive (furthest in same class) and hardest negative
(closest in different class) and applies margin ranking loss.
"""
function triplet_loss(emb::Array{Float64,2}, labels::Vector{Int}; margin::Float32=0.5f0)
    D = distances(emb)    

    B = length(labels)

    loss = 0f0
    count = 0

    for i in 1:B
        same = Array(labels .== labels[i]) .& (1:B .!= i)   # mask for positives (same class) except for anchor
        diff = Array(labels .!= labels[i])   # mask for negatives (different class) 

        if !any(same) || !any(diff)
            continue                
        end

        hardest_pos = maximum(D[i, same])   # furthest positive in batch
        hardest_neg = minimum(D[i, diff])   # closest negative in batch

        l = max(0f0, hardest_pos - hardest_neg + margin)  # batch-hard triplet loss
        loss += l
        count += 1
    end

    return count > 0 ? loss / count : 0f0  # average over valid anchors
end

