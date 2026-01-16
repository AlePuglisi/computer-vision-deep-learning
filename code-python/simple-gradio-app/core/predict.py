# prediction function module, used in the main app file 

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os 

class CustomCNNModel(nn.Module):
    # always in a NN define
    # input dim as image input size (depends on the data loader as 128x128)
    # num_classes also define on the dataset (we have 3 classes)
    def __init__(self, input_dim, num_classes):
        super(CustomCNNModel, self).__init__() # Initialize base model 
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            # Conv1
            # in conv2d you specify the filter channels number 
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # RGB require 3 channels, so always match input 
            nn.BatchNorm2d(32), # apply batch normalization on all features 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # half of original input 

            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), # apply batch normalization on all features 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # half of original input  

            # Conv3
            # in conv2d you specify the filter channels number 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128), # apply batch normalization on all features 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # half of original input  

            # Conv4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), # apply batch normalization on all features 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # half of original input  
            # doubling the output size at each step 

            # the output of these convolutional blocks is required to know the size of next layer 
        )

        self._to_linear = self._get_conv_output(self.input_dim)

        # create the fully connected layers as sequential network 
        self.fc_layers = nn.Sequential(
            # Use the extracted features in a fully connected layers MLP, to train model on the patterns of menaingful features 
            nn.Linear(self._to_linear, 512), # Fully connected 
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, self.num_classes), 
        )

    def _get_conv_output(self,input_dim=128):
        with torch.no_grad(): # feed forward the conv layers withou tgradient calculation
            # create a dummy image
            dummy_input = torch.zeros(1, 3, input_dim, input_dim) # batch size of 1, 3xwxh
            dummy_output = self.conv_layers(dummy_input)

            # we want the length of the flatten output 
            to_linear = dummy_output.view(1, -1).size(1) # reshape as row vector 
            # initialize to_linear as number of required neurons in the first FC layer 
            return to_linear


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # reshape in forward propagate, flatten for the FC layers
        x = self.fc_layers(x)
        return x
    # after the forward the argmax is extracted to get the class

class ImageClassifier():

    # loading model architecture 
    # ( No need of data loader since it will be used just as inference, not training in batches)
    # Using the code from a previous ipynb code on custom CNN 
    def __init__(self, model_path, class_name=None):
        # first set the device used by our code based on GPU availability 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # steps...
        # 1. Define CNN architecture 
        # 2. Load CNN architecture train wieghts 
        # 3. Create index to label map
        # 4. Define transformations required from any input image, to 
        #    Resize, conversion, normalization required in any input image to adapt to the expected image size 

        # CNN Architecture
        ## CNN()
        self.model = CustomCNNModel(input_dim=128, num_classes=3).to(self.device)

        # load pth to initialzie CNN weights
        ## CNN.load_dict()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)) # load model to device 
        self.model.eval() # initialize as evaluation mode 

        # Index to label map
        if class_name is None:
            self.class_name = {0: 'Cat', 1:'Dog', 3:'person'}
        else: 
            self.class_name = class_name

        # Tranformation input 
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), # lower resolution for fatser speed 
            transforms.ToTensor(), # instead of pillow image, pytorch model expect tensors 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize in the range -1, 1
        ])

    # take image as input and output the class predicted 

    def predict(self, image_path):
        # Load image with Pillow (as expected by pytorch)
        # Then the pillow image can be used and transformed 

        # perform prediction with the loded model 

        # label map to map the output ot specific classes

        # write text to the input image using opencv (modify input image with predicted class
        
        # return class, output image (to show on the gradio app)
        image = Image.open(image_path).convert("RGB")
        # Not suitable for model: 
        image_tensor = self.transform(image).unsqueeze(0).to(self.device) # unsqueeze the image 

        # send image tensor for prediction 
        with torch.no_grad():
            output = self.model(image_tensor)
            # output as 2D tensor [[0.3, 0.7, 0.9]]
            # max give value, idx (we get the index)
            _, predicted = torch.max(output, 1)

        label = self.class_name[predicted.item()]

        img = cv2.imread(image_path)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        output_path = "labeled_image.jpg"
        # temporary save the labeled image
        cv2.imwrite(output_path, img)

        cwd = os.getcwd()
        output_path = os.path.join(cwd, output_path)

        return label, output_path


