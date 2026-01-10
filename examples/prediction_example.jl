using Statistics: mean

include("../src/dataset_loading.jl")
include("../src/model_save_load.jl")
include("../src/hybrid_model.jl")
include("../src/visualisation.jl")

function prediction_example(path_dataset::String = "data/test", path_model::String="model/cnn_metric.bson")
    model = get_defualt_hybrid_model()
    load_model!(path_model, model)

    imgs, labels = load_test(path_dataset)

    embs = get_embs(model, imgs)
    pred = get_class(model, imgs)

    acc = mean(pred .== labels)

    print("acc: ")
    println(acc)

    visualise2D(embs, labels)
    visualise3D(embs, labels)
end