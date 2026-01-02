function l2_normalize(x; dims=1)
    norms = sqrt.(sum(x.^2, dims=dims))
    norms = max.(norms, 1e-6)      
    return x ./ norms
end

"""
Compute combined metric loss (triplet + contrastive) on a batch of embeddings.

# Arguments
- `emb::Array{Float32,2}`: embeddingy batchu, shape (embedding_dim, batch_size)
- `y::Vector{Int}`: labels
- `use_triplet::Bool`: whether to include triplet loss (default: true)
- `use_contrastive::Bool`: whether to include contrastive loss (default: true)
- `margin_triplet::Float32`: margin for triplet loss (default: 0.5)
- `margin_contrastive::Float32`: margin for contrastive loss (default: 1.0)

# Returns
- `loss::Float32`: scalar loss combining requested metric losses
"""
function metric_loss(emb::Array{Float32,2}, y::Vector{Int};
        use_triplet::Bool = true,
        use_contrastive::Bool = true,
        margin_triplet::Float32 = 0.5f0,
        margin_contrastive::Float32 = 1.5f0,
        triplet_perc::Float32 = 0.5f0
    )

    loss = 0f0

    if use_triplet
        loss +=  triplet_perc * triplet_loss(emb, y; margin=margin_triplet)
    end

    if use_contrastive
        loss += contrastive_loss(emb, y; margin=margin_contrastive)
    end

    return loss
end


"""
Compute pairwise L2 distances between all embeddings in the batch.

# Arguments
- `emb::Array{Float32,2}`: embeddingy batchu, shape (embedding_dim, batch_size)

# Returns
- `D::Array`: B×B matrix of pairwise Euclidean distances
"""
function distances(emb::Array{Float32,2})
    sq = sum(emb.^2, dims=1)               
    dist2 = sq .+ sq' .- 2.0 .* (emb' * emb)
    dist2 = max.(dist2, 1e-12)
    return sqrt.(dist2)
end


"""
Compute contrastive loss on a batch of embeddings.

# Arguments
- `emb::Array{Float32,2}`: embeddings, shape (embedding_dim, batch_size)
- `labelsy::Vector{Int}`: labels
- `margin::Float32`: distance margin for negative pairs (default: 1.0)

# Returns
- `loss::Float32`: scalar contrastive loss

Positive pairs are pulled together; negative pairs are pushed apart beyond the margin.
"""
function contrastive_loss(emb::Array{Float32,2}, labels::Vector{Int}; margin::Float32=1.0f0)
    D = distances(emb)  
    
    pos_mask = labels .== labels' # matrix of positive doubles
    neg_mask = .!pos_mask # matrix of negative embedings doubles

    pos_loss = D[pos_mask] .^ 2 # minimizing distances of same-labeled embeddings
    neg_loss = max.(0f0, margin .- D[neg_mask]) .^ 2 # maximizing distances of differently labeled embeddings

    sum_pos_loss = sum(pos_loss) / length(pos_loss)
    sum_neg_loss = sum(neg_loss) / length(neg_loss)

    return sum_pos_loss + sum_neg_loss

end


"""
Compute batch-hard triplet margin loss on a batch of embeddings.

# Arguments
- `emb::Array{Float32,2}`: embeddings, shape (embedding_dim, batch_size)
- `labelsy::Vector{Int}`: labels
- `margin::Float32`: margin for triplet loss (default: 0.5)

# Returns
- `loss::Float32`: scalar triplet loss

For each anchor, selects hardest positive (furthest in same class) and hardest negative
(closest in different class) and applies margin ranking loss.
"""
function triplet_loss(emb::Array{Float32,2}, labels::Vector{Int}; margin::Float32=0.5f0)
    D = distances(emb)    

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

        l = max(0f0, hardest_pos - hardest_neg + margin)  # batch-hard triplet loss
        loss += l
        count += 1
    end

    tri_loss = count > 0 ? loss / count : 0f0  # average over valid anchors

    return tri_loss

end

