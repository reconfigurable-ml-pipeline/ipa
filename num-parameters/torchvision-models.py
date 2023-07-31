from torchvision import models

available_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

num_params = {}

# Instantiate the ResNet model
for model_name, model_variant in available_models.items():
    model = available_models[model_name]()

    # Count the number of parameters
    num_params[model_name] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


print(50 * "-" + " results " + 50 * "-")
for model_name, num_params in num_params.items():
    print(f"Number of parameters in model, {model_name} is: {num_params/(10**6)} M")
