import torch
import torchvision.models as models
import matplotlib.pylab as plt

from PIL import Image
from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *


if __name__ == "__main__":
    # ++++++++++++++++++++ Cityscapes Visualisation +++++++++++++++++++++++++
    data_path = 'dataset/cityscapes'
    im_size = [512, 1024]
    num_segments = 19
    test_idx = get_cityscapes_idx(data_path, train=False)

    device = torch.device("cpu")
    model = DeepLabv3Plus(models.resnet101(), num_classes=num_segments).to(device)
    colormap = create_cityscapes_label_colormap()

    # visualise image id 61 in validation set
    im_id = 61
    im = Image.open('dataset/cityscapes/images/val/{}.png'.format(im_id))
    gt_label = Image.fromarray(cityscapes_class_map(Image.open('dataset/cityscapes/labels/val/{}.png'.format(im_id))))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0),augmentation=False)


    model.load_state_dict(torch.load('model_weights/cityscapes_label20_sup.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_sup = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_sup[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/cityscapes_label20_semi_classmix.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_classmix = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_classmix[label_tensor == -1] = -1

    model.load_state_dict(torch.load('model_weights/cityscapes_label20_semi_classmix_reco.pth', map_location=torch.device('cpu')))
    model.eval()
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    label_reco[label_tensor == -1] = -1

    fig = plt.figure(figsize=(10, 10))

    gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)), alpha=.7)
    sup_blend = Image.blend(im, Image.fromarray(color_map(label_sup[0].numpy(), colormap)), alpha=.7)
    classmix_blend = Image.blend(im, Image.fromarray(color_map(label_classmix[0].numpy(), colormap)), alpha=.7)
    reco_blend = Image.blend(im, Image.fromarray(color_map(label_reco[0].numpy(), colormap)), alpha=.7)

    ax0 = plt.subplot(2, 2, 1)
    plt.imshow(gt_blend)
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    ax0.set_xlabel('Ground Truth')
    ax1 = plt.subplot(2, 2, 2)
    plt.imshow(sup_blend)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xlabel('Supervised')
    ax2 = plt.subplot(2, 2, 3)
    plt.imshow(classmix_blend)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlabel('ClassMix')
    ax3 = plt.subplot(2, 2, 4)
    plt.imshow(reco_blend)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xlabel('ClassMix + ReCo')
    plt.show()