import torch
from torchvision import transforms
from PIL import Image
from model import UAVClassifier

# Load model and set to evaluation mode
model = UAVClassifier(num_classes=3)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define class names based on your data organization
class_names = ['airplane', 'drone', 'helicopter']  # Adjust as needed

def predict_image(image_path):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class

if __name__ == "__main__":
    image_path = 'C:\\Users\\KHALED\\Downloads\\image_12345 (2).jpg'  # Update with your image path
    result = predict_image(image_path)
    print(f"The predicted class is: {result}")
