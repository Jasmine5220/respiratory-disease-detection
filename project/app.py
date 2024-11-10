# Import necessary libraries
from flask import Flask, request, render_template, redirect, url_for
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model (make sure to modify the last layer)
model = models.resnet18(pretrained=False)  # Load the base ResNet model (not pretrained)

# Modify the last fully connected layer to match the number of classes (4 classes in this case)
model.fc = torch.nn.Linear(model.fc.in_features, 4)

# Load the saved model weights (ensure it's loaded into the modified model)
model.load_state_dict(torch.load('model.pth'))

model.eval()

# Define transformations (adjust based on the model's input requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

# Sample data for testing metrics (use actual test data)
y_true = [0, 1, 1, 0, 1]  # Ground truth labels
y_pred = [0, 1, 0, 0, 1]  # Model's predictions on the test set

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Route for uploading an image and getting a prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded."
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file."
        
        # Save the file to the uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Open and convert the image to RGB format
        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations and add batch dimension
        input_tensor = transform(image).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        # Calculate model metrics (can be customized for this specific image or test data)
        accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
        
        # Correct class label mapping
        class_names = ['Normal', 'Viral Pneumonia', 'COVID-19', 'Lung Opacity']
        prediction_class = class_names[prediction]  # Match the predicted index to the class label

        # Debugging: Print the predicted class
        print(f"Predicted Class Index: {prediction}")
        print(f"Predicted Class Name: {prediction_class}")

        # Render the prediction page with results
        return render_template('result.html', prediction_class=prediction_class,
                               accuracy=accuracy, precision=precision,
                               recall=recall, f1=f1)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
