__precompile__(false)
module GalaxyCNN

using UMAP
using Plots

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


export augment!

export split

export load_batch

export loss_fn, train_model

export load_train, load_test

export get_default_hybrid_model, get_probs, get_embs, get_class

export metric_loss

export save_model, load_model!

export show_image

export visualise2D, visualise3D

end
