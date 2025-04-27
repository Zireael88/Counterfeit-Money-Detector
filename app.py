from flask import Flask, render_template, request
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.functional import softmax
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Real and Fake
model.load_state_dict(torch.load('resnet18_counterfeit.pth', map_location=torch.device('cpu')))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict
        image = Image.open(filepath).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        output = model(image)

        # Get probabilities using softmax
        probabilities = softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # Swap the result: Counterfeit becomes Real and vice versa
        label = 'Real Money' if predicted.item() == 1 else 'Counterfeit Money'
        confidence_score = confidence.item() * 100  # Convert to percentage

        return render_template('index.html', filename=file.filename, label=label, confidence=confidence_score)

    return render_template('index.html')

# To show uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
