"""Functionality for OCR based timestamp detection"""
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Normalize, Resize

IMAGE_TRANSFORM = Compose([ToTensor(), Resize((16, 25), antialias=True), Normalize((0.1307,), (0.3081,))])



def get_timestamp_area_tapo():
    """Returns a tuple of (start_x, end_x, start_y, end_y) for the timestamp area"""
    return (350, 605, 0, 50)

class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(60, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), 60)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

    @staticmethod
    def extract_time_digits_tapo(frame):
        """Extracts digits from an image. Parameters are specific to TAPO TP-link camera.
        returns a list of length 6, with [h,h,m,m,s,s]"""
        start_x, end_x, start_y, end_y = get_timestamp_area_tapo()
        digit_x = list(np.linspace(start_x , end_x, 9))
        digits = [
            frame[:end_y, int(x1):int(x2), :] for x1, x2  in zip(digit_x, digit_x[1:])
        ]
        return digits[0:2] + digits[3:5] + digits[6:8]

    def preprocess_images(self, images):
        return torch.stack([IMAGE_TRANSFORM(image) for image in images])

    def predict(self, images):
        self.eval()
        with torch.no_grad():
            logits = self(self.preprocess_images(images))
            return torch.argmax(logits, dim=1)
    
    def get_timestamp(self, frame):
        digits = self.extract_time_digits_tapo(frame)
        predictions = self.predict(digits).tolist()
        hour = predictions[0] * 10 + predictions[1]
        minute = predictions[2] * 10 + predictions[3]
        second = predictions[4] * 10 + predictions[5]
        timestamp = datetime.datetime.now()
        # update time
        return timestamp.replace(hour=hour, minute=minute, second=second)