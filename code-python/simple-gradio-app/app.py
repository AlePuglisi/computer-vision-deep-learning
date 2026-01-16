from tkinter import Image
import gradio as gr 
import os 
from core.predict import ImageClassifier
from PIL import Image 

cwd = os.getcwd()
model_path = os.path.join(cwd, "model", "cnn_128_model-100.pth")

classifier = ImageClassifier(model_path=model_path, class_name=None)



# predict function will use the cnn model to predict the image deployed on the web properly
def classify_image(image):
    image_path = "uploaded_image.jpg"
    # temporary save the uploaded image
    image.save(image_path)

    label, output_path = classifier.predict(image_path)

    return label, Image.open(output_path)

    
demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type='pil'),
    outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Labeled Image")],
    title="Image Classification Gradio app",
    description="Upload an Image to classify it as Dog, Cat or person"
)

if __name__ == "__main__":
    demo.launch()

# running this python script it will run locally on your device, not online for others...