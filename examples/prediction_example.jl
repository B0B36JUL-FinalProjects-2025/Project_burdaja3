using Statistics: mean
using Flux: onecold
using GalaxyCNN

"""
    prediction_example(path_dataset::String = "data/test",
                       path_model::String = "model/cnn_metric.bson")

Run a prediction and visualisation example of embeddings using a pretrained hybrid model.

# Arguments
- `path_dataset::String`: path to the test dataset
- `path_model::String`: path to the saved model
"""
function prediction_example(path_dataset::String = "data/test", path_model::String="model/cnn_metric.bson")
    model = get_default_hybrid_model()
    load_model!(path_model, model)

    imgs, labels = load_test(path_dataset)

    imgs = (Float32.(imgs)) ./ 255f0

    embs = get_embs(model, imgs)
    pred = get_class(model, imgs)

    acc = mean(pred .== labels)

    print("acc: ")
    println(acc)

    visualise2D(embs, labels)
    visualise3D(embs, labels)

    return
end