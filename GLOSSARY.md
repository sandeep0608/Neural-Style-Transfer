# Glossary - Neural Style Transfer

Plain-English definitions of every term used in this project, roughly in the order
you meet them. Written for someone new to deep learning.

## Core idea

**Neural style transfer** - Taking the *content* of one image (a photo) and
re-rendering it in the *style* of another image (a painting), using a neural
network.

**Content** - *What* is in an image and *where*: the objects and their arrangement.
A face, a building, a river. Position matters for content.

**Style** - The look and feel independent of position: textures, brushstrokes,
color palettes. A painting's "swirliness" is style; it appears everywhere, so
position does *not* matter for style.

## Images as numbers

**Tensor** - A grid of numbers. In PyTorch, images are tensors. Our images have the
shape `[1, 3, H, W]`: a batch of 1 image, 3 color channels (R, G, B), H pixels tall,
W pixels wide.

**Channel** - One of the stacked layers of a tensor. A color image has 3 channels
(red, green, blue). Deeper in the network, "channels" instead mean *feature
detectors* (see Feature map).

**Batch dimension** - The leading `1` in `[1, 3, H, W]`. Neural networks expect a
*batch* of images at once, so even a single image is presented as "a batch of one."

**Normalization** - Adjusting pixel values using fixed average numbers (mean) and
spread numbers (std) so they match what the network was trained on. VGG expects
inputs normalized with the **ImageNet mean/std**; feeding un-normalized images
gives garbage results.

**ImageNet mean / std** - The specific average and spread values
(`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`) computed from the
ImageNet dataset that VGG was trained on. We normalize with these, and reverse them
to view results.

**Un-normalize** - Reversing normalization to get a viewable image: multiply by std,
then add mean (the reverse operations, in reverse order).

**Clamp** - Forcing numbers to stay within a range (here, `[0, 1]`) so small math
errors don't push pixel values out of the valid range.

## The network

**CNN (Convolutional Neural Network)** - A neural network designed for images. It
processes an image through a stack of layers, detecting simple features early
(edges) and complex features later (objects).

**VGG-19** - A specific, well-known CNN with 19 main layers, trained on ImageNet to
recognize objects. We borrow its "vision" as a fixed feature extractor.

**Pretrained** - Already trained by someone else. We download VGG-19's learned
weights rather than training from scratch.

**Convolutional layer (`Conv2d`)** - A layer that slides small filters over the
image to detect features. Each filter produces one feature map.

**Pooling layer (`MaxPool2d`)** - A layer that shrinks the image (halves height and
width here), letting deeper layers see larger patterns. The paper suggests average
pooling for smoother results.

**ReLU** - A simple step applied after each conv layer: keep positive values, set
negatives to zero. We tap the conv layers, not the ReLUs.

**Classifier** - The final part of VGG that outputs "this is a dog." We discard it;
we only want the internal feature layers.

**Feature map / activations** - The output of a layer: the transformed version of
the image at that depth. Shallow layers' feature maps look like edges and textures;
deep layers' feature maps represent high-level content.

**Layer names (`conv4_2`, `conv1_1`, ...)** - The paper's way of naming layers:
`conv<block>_<position>`. Blocks are separated by pooling layers. We map these names
to the numbered layers VGG actually exposes.

**Freeze / `requires_grad_(False)`** - Turning off learning for the network's
weights so they never change. We use VGG only as a fixed tool.

**`eval()` mode** - Puts a model into "using it," not "training it," mode. Standard
when you're not updating weights.

## The losses

**Loss** - A single number measuring how wrong the current result is. Lower is
better. The whole process pushes the loss down.

**Mean Squared Error (MSE)** - A way to measure difference: subtract, square each
difference (so negatives don't cancel and big errors count extra), then average.
Zero when two things are identical.

**Content loss** - MSE between the generated image's `conv4_2` feature map and the
content photo's `conv4_2` feature map. Measures "is the right stuff in the right
place?"

**Gram matrix** - A `channels x channels` grid measuring how much each pair of
feature channels fires together across the whole image. It captures style because
it records *which patterns co-occur* while discarding *where* they occur. Built by
flattening the feature maps and multiplying them by their own transpose.

**Transpose** - Flipping a matrix's rows and columns. Used when computing the Gram
matrix (`features x features^T`).

**Style loss** - MSE between the Gram matrices of the generated image and the style
image, summed across the five style layers (each weighted 1/5).

**Total loss** - `content_weight x content_loss + style_weight x style_loss`. The
single number we minimize.

**content_weight / style_weight (alpha and beta)** - Dials that balance the two
goals. More style_weight = more painterly; more content_weight = more like the
photo. The paper uses an alpha/beta ratio around 1e-3 to 1e-4.

## The optimization

**Optimization** - Repeatedly adjusting something to reduce a loss. Here, we adjust
the generated image's pixels.

**Gradient** - The "which direction, and how much, should I change this to reduce
the loss?" signal, computed by calculus. Short for the slope of the loss.

**`requires_grad_(True)`** - Turning gradient tracking *on* for the generated image,
marking it as the thing we optimize. (The opposite of freezing.)

**Backpropagation / `backward()`** - The algorithm that computes gradients by
working backward through the network.

**`detach()`** - Telling PyTorch "don't track this for gradients; I'm just looking."
Used when saving/viewing a result.

**Optimizer** - The algorithm that uses gradients to update whatever is being
optimized. Here it updates pixels.

**L-BFGS** - The specific optimizer the paper uses, well suited to optimizing an
image. It needs its loss computation wrapped in a **closure**.

**Closure** - A small function that computes and returns the loss. L-BFGS calls it
multiple times per step, which is why the loop looks slightly different from typical
PyTorch training.

**Step / iteration** - One round of: compute loss, compute gradients, nudge the
pixels. We run a few hundred.

## Practical terms

**Virtual environment (`venv`)** - An isolated box for a project's libraries so they
don't conflict with other projects, and so the setup is reproducible.

**`unsqueeze` / `squeeze`** - Adding (`unsqueeze`) or removing (`squeeze`) a
dimension from a tensor. We add the batch dimension on the way in and remove it on
the way out.

**`interpolate`** - Resizing a tensor. We use it to make the style image match the
content image's dimensions so the content loss lines up.

**Shape mismatch** - The most common deep-learning error: trying to do math on two
tensors of different sizes. The fix is to make their shapes match.
