import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)
        self.output_fc = nn.Linear(3000, output_dim)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = torch.sigmoid(self.input_fc(x))
        output = self.output_fc(h)

        return output, h


def get_prediction(image_bytes):

    tensor = transform_image(image_bytes=image_bytes)
    y_pred, _ = model(tensor)
    y_prob = F.softmax(y_pred, dim=-1)
    prob, label_index = y_prob.max(1)

    return label_index.item(), prob.item()


def transform_image(image_bytes):

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                                ])
    image = Image.open(io.BytesIO(image_bytes))

    return transform(image)


def mi_face(label_index, num_iterations, gradient_step):

    criterion = torch.nn.CrossEntropyLoss()
    tensor = torch.zeros(112, 92).unsqueeze(0)
    image = tensor
    tensor.requires_grad = True
    min_loss = float("inf")

    for i in range(num_iterations):
        pred, _ = model(tensor)
        loss = criterion(pred, torch.tensor([label_index]))
        loss.backward()
        with torch.no_grad():
            # tensor = torch.clamp(tensor - gradient_step * tensor.grad, 0, 255)
            tensor = (tensor - gradient_step * tensor.grad)
            if loss < min_loss:
                min_loss = loss
                image = tensor
        tensor.requires_grad = True
        print(min_loss)

    return image


if __name__ == '__main__':

    model = torch.load('atnt-mlp-model.pt')
    model.eval()

    # random generated dummy names
    class_index = json.load(open('class_index.json'))

#    with open('data_pgm/faces/s01/1.pgm', 'rb') as f:
#        image_bytes = f.read()
#        label_index, prob = get_prediction(image_bytes=image_bytes)
#        label = class_index[str(label_index)]
#        print(label, prob)

    image = mi_face(4, 30, 0.1)

    plt.imshow(image.permute(1, 2, 0).detach().numpy())
    plt.show()

