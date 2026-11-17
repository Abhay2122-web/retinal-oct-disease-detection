"""Grad-CAM utilities for supported OCT classification architectures.

Uses a tensor gradient hook rather than a module full-backward hook to avoid
PyTorch view/in-place modification conflicts with in-place ReLU layers.
"""

import torch
import torch.nn.functional as F


def find_last_conv_layer(model, model_name):
    """Return the feature layer used to generate the Grad-CAM heatmap."""
    name = model_name.lower()

    if name == "custom_cnn":
        return model.features[-3]
    if name == "resnet18":
        return model.layer4[-1]
    if name == "vgg16":
        return model.features[-1]
    if name == "mobilenet_v2":
        return model.features[-1]
    if name == "densenet121":
        return model.features.denseblock4

    raise ValueError(f"Grad-CAM is not configured for model: {model_name}")


def make_gradcam(model, image_tensor, model_name, target_class=None):
    """Return (heatmap, predicted_class_index, class_probabilities)."""
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [1, C, H, W].")

    activations = []
    gradients = []
    target_layer = find_last_conv_layer(model, model_name)

    def forward_hook(module, inputs, output):
        # Clone activations because subsequent in-place layers may modify output.
        activations.append(output.detach().clone())

        # Tensor hooks avoid the view handling of module full-backward hooks.
        if output.requires_grad:
            output.register_hook(
                lambda grad: gradients.append(grad.detach().clone())
            )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(image_tensor)

            class_id = (
                int(logits.argmax(dim=1).item())
                if target_class is None
                else int(target_class)
            )

            if class_id < 0 or class_id >= logits.shape[1]:
                raise ValueError(
                    f"target_class must be between 0 and {logits.shape[1] - 1}."
                )

            logits[0, class_id].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM could not capture activations or gradients.")

        acts = activations[0]
        grads = gradients[0]

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam_tensor = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_tensor = F.interpolate(
            cam_tensor,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam_tensor[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        probabilities = torch.softmax(
            logits.detach(), dim=1
        )[0].cpu().numpy()

        return cam, class_id, probabilities

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
