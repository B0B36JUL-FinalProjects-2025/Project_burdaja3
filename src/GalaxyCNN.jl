__precompile__(false)
module GalaxyCNN

include("augmentation/augment.jl")

include("dataset_creation/galaxies_split.jl")

include("batches.jl")
include("cnn.jl")
include("hybrid_model.jl")
include("metric.jl")
include("dataset_loading.jl")
include("model_save_load.jl")
include("show_image.jl")
include("visualisation.jl")


export augment!, translate!, noise!, zoom!, rotate!

export split

export load_batch, load_test

export loss_fn, train_model

export load_train, load_test

export HybridModel, get_default_hybrid_model, get_probs, get_embs, get_class

export metric_loss_fn, l2_normalize, distances, TripletLoss, MetricLoss, ContrastiveLoss

export save_model, load_model!

export show_image

export visualise2D, visualise3D

end
