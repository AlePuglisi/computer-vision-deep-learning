# prediction function module, used in the main app file 

import torch 
# 8 min tutorial
class ImageClassifier():

    # loading model architecture 
    # ( No need of data loader since it will be used just as inference, not training in batches)
    # Using the code from a previous ipynb code on custom CNN 
    def __init__(self):
        # first set the device used by our code based on GPU availability 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define CNN architecture 
        # Load CNN architecture train wieghts 
        # Create index to label map
        # Define transformations required from any input image
        # resize, conversion, normalization required in any input image to adapt to the expected image size 

        # CNN Architecture
        ## CNN()
        ## CNN.load_dict()
        # Tranformation input 

        pass

    # take image as input and output the class predicted 

    def predict(self):
        # Load image with Pillow (as expected by pytorch)
        # Then the pillow image can be used and transformed 

        # perform prediction with the loded model 

        # label map to map the output ot specific classes

        # write text to the input image using opencv (modify input image with predicted class
        
        # return class, output image (to show on the gradio app)
        pass