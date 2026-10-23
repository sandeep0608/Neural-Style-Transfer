import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Maps VGG-19 layer numbers (as printed) to the paper's layer names.
# conv4_2 is for CONTENT. The other five are for STYLE.
LAYER_MAP = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",
    "28": "conv5_1",
}

# The 5 style layers from the paper, each weighted equally (1/5).
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
STYLE_WEIGHTS = {layer: 0.2 for layer in STYLE_LAYERS}


def load_image(path, max_size=384):
    image = Image.open(path).convert("RGB")

    width, height = image.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


def tensor_to_image(tensor, save_path):
    tensor = tensor.clone().detach().squeeze(0)

    for channel, mean, std in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        channel.mul_(std).add_(mean)

    tensor = tensor.clamp(0, 1)

    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.save(save_path)
    print("Saved image to:", save_path)


def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    vgg.eval()

    return vgg


def get_features(image, vgg, layer_map=LAYER_MAP):
    features = {}
    x = image

    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layer_map:
            features[layer_map[name]] = x

    return features


def content_loss(generated_features, content_features):
    loss = torch.mean((generated_features - content_features) ** 2)
    return loss


def gram_matrix(feature_map):
    batch, channels, height, width = feature_map.size()
    features = feature_map.view(channels, height * width)
    gram = torch.mm(features, features.t())
    gram = gram / (channels * height * width)
    return gram


def style_loss(generated_features, style_grams, style_layers, weights):
    loss = 0.0
    for layer in style_layers:
        generated_gram = gram_matrix(generated_features[layer])
        target_gram = style_grams[layer]
        layer_loss = weights[layer] * torch.mean((generated_gram - target_gram) ** 2)
        loss = loss + layer_loss
    return loss


def run_style_transfer(content, style, vgg, steps=300, style_weight=1e6, content_weight=1.0):
    # 1. Precompute the targets (these never change during optimization).
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}
    content_target = content_features["conv4_2"]

    # 2. Our generated image STARTS as a copy of the content photo.
    generated = content.clone().requires_grad_(True)

    # 3. The optimizer adjusts the IMAGE (generated), not the network.
    optimizer = torch.optim.LBFGS([generated])

    run = [0]

    while run[0] <= steps:

        def closure():
            optimizer.zero_grad()

            generated_features = get_features(generated, vgg)

            c_loss = content_loss(generated_features["conv4_2"], content_target)
            s_loss = style_loss(generated_features, style_grams, STYLE_LAYERS, STYLE_WEIGHTS)

            total = content_weight * c_loss + style_weight * s_loss
            total.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}:  content {c_loss.item():.2f}   style {s_loss.item():.6f}")

            return total

        optimizer.step(closure)

    return generated


# --- RUN THE STYLE TRANSFER ---
vgg = load_vgg()
OUTPUT_SIZE = 512
content = load_image("content.jpg")
style = load_image("style.jpg")

# Resize style to match content's dimensions (so content loss lines up).
style = F.interpolate(style, size=content.shape[2:])

print("Starting style transfer. On CPU this takes a few minutes, be patient...")
output = run_style_transfer(content, style, vgg, steps=300)

tensor_to_image(output, "result.jpg")
print("Done! Open result.jpg")