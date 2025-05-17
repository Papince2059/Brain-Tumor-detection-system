# from flask import Flask, render_template, request, redirect
# import os
# from gradcam_utils import preprocess_image, generate_gradcam, overlay_heatmap
# import torch
# import torch.nn as nn
# from torchvision import models
# from torchvision import transforms
# from flask import render_template
# from gem import fetch_definition_data

# APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(pretrained=False)
# num_classes = 4  # Assuming 4 classes
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.load_state_dict(torch.load('model/brain_tumor_resnet50.pth', map_location=device))
# model.to(device)
# model.eval()

# tumor_classes = ['Meningioma', 'Glioma', 'Pituitary Tumor', 'No Tumor']

# # Image size constant
# IMG_SIZE = 224

# # Define transformation for the image
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'mri_image' not in request.files:
#         return redirect('/')
    
#     file = request.files['mri_image']

#     if file.filename == '':
#         return redirect('/')

#     if file:
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         img_tensor, pil_image, np_image = preprocess_image(filepath, transform)

#         # Grad-CAM generation
#         target_layer = model.layer4[-1]
#         cam, pred_class = generate_gradcam(model, img_tensor, np_image, target_layer, device)

#         predicted_class = tumor_classes[pred_class]

#         # Save Grad-CAM heatmap
#         heatmap_filename = 'gradcam_' + filename
#         heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
#         overlay_heatmap(cam, filepath, heatmap_path)

#         # Pass relative URLs to HTML
#         relative_path = os.path.join('static', 'uploads', filename)
#         heatmap_relative_path = os.path.join('static', 'uploads', heatmap_filename)

#         return render_template('result.html',
#                             prediction=predicted_class,
#                             original_image=relative_path,
#                             heatmap_image=heatmap_relative_path)
    
#     return redirect('/')
#     # Replace return jsonify(result) with:
#     # return render_template("/home/tanuj/Brain_MRI/templates/result.html", predicted_class=tumor_classes[pred_class])




# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect
import os
from gradcam_utils import preprocess_image, generate_gradcam, overlay_heatmap
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from flask import render_template
from gem import fetch_definition_data  # Importing the function from gem.py

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_classes = 4  # Assuming 4 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model/brain_tumor_resnet50.pth', map_location=device))
model.to(device)
model.eval()

tumor_classes = ['Meningioma', 'Glioma', 'Pituitary Tumor', 'No Tumor']

# Image size constant
IMG_SIZE = 224

# Define transformation for the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'mri_image' not in request.files:
        return redirect('/')

    file = request.files['mri_image']

    if file.filename == '':
        return redirect('/')

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_tensor, pil_image, np_image = preprocess_image(filepath, transform)

        # Grad-CAM generation
        target_layer = model.layer4[-1]
        cam, pred_class = generate_gradcam(model, img_tensor, np_image, target_layer, device)

        predicted_class = tumor_classes[pred_class]

        # Fetch disease information from the gem.py function
        disease_info = fetch_definition_data(predicted_class)

        # Save Grad-CAM heatmap
        heatmap_filename = 'gradcam_' + filename
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        overlay_heatmap(cam, filepath, heatmap_path)

        # Pass relative URLs and fetched disease info to HTML
        relative_path = os.path.join('static', 'uploads', filename)
        heatmap_relative_path = os.path.join('static', 'uploads', heatmap_filename)

        return render_template('result.html',
                               prediction=predicted_class,
                               original_image=relative_path,
                               heatmap_image=heatmap_relative_path,
                               disease_info=disease_info)  # Pass the fetched information

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

