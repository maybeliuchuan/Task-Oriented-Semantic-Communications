"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
# from torchsummary import summary
from torchvision import models, transforms
import torch.nn as nn
from torch.autograd import Variable

# from misc_functions import save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for layer, module in self.model._modules.items():
            x = module(x)  # Forward
            if layer == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
                return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        # channel
        snr = 25    #信噪比
        snr = 10**(snr/10.0)  
        xpower = torch.sum(x**2,1)/512.
        npower = xpower/snr
        noise = torch.FloatTensor(512,x.size(0))
        noise = noise.normal_()*torch.sqrt(npower)
        noise = noise.transpose(1,0)
        x = x + noise
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print("conv_output shape:", conv_output.shape, model_output.shape)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        print("guided_gradients shape: ",guided_gradients.shape)
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        print("target shape: ",target.shape)
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        print(np.max(weights), np.min(weights))
        print("weights shape: ",weights.shape)
        print("weights sort: ", weights.argsort()[::-1])
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        print("cam shape: ",cam.shape)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        print(np.max(cam), np.min(cam))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam


if __name__ == '__main__':
    chkpt_path = 'E:/机器学习/pytorch-cnn-visualizations-master/src/classification/MI+Classification/Resnet18/resnet_10_classes_25db_mi_12.pth'
    img_path = 'E:/机器学习/datasets/STL10/test/truck/17.png'

    classes = 10
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    model.load_state_dict(torch.load(chkpt_path, map_location='cpu'))
    # print(model)
    # summary(model, (3, 128, 128)) 
    model.eval()

    image = Image.open(img_path)
    resize_obj = transforms.Resize(256)
    original_image = resize_obj(image)
    crop_obj = transforms.CenterCrop((224, 224)) 
    crop_image = crop_obj(original_image)
    original_image = crop_image
    tensor_obj = transforms.ToTensor()
    prep_img = tensor_obj(crop_image)
    prep_img.unsqueeze_(0)
    # image_np = np.array(crop_image)

    target_class = 9

    y = model(prep_img)
    print(y.data)
    print(y.data.numpy().argsort()[0][-1])

    layer_name = 'layer4'
    file_name_to_export = 'cat23'

    # Grad cam
    grad_cam = GradCam(model, layer_name)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    # save_class_activation_images(original_image, cam, file_name_to_export)
    # save_class_activation_images(image_np, cam, file_name_to_export)

    print('Grad cam completed')
