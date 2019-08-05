import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import load_data_one_pytorch as load_data

def prepare_data():
    return (np.array(load_data.train_x), np.array(load_data.train_y),
                np.array(load_data.test_x), np.array(load_data.test_y))


x_train, y_train, x_test, y_test = prepare_data()

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 48
batch_size = 100
learning_rate = 0.001


def evalution():
    correct = 0
    total = 0
    x_test_tensor = torch.from_numpy(x_test).float()
    y_test_tensor = torch.from_numpy(y_test)

    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    answer = np.array([np.argmax(i) for i in y_test])
    predict = predicted.numpy()

    acc = np.sum(predict == answer) / len(predict)
    print('Single phone test accuracy: {:.2%}'.format(acc))
    print('----------------------------------\n')


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=48):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*8*32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.dropout(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

model = ConvNet(num_classes).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = int(len(x_train)/batch_size -  1)
for epoch in range(num_epochs):
    print("epoch ", epoch)
    for i in range(total_step):
        local_x = x_train[i*batch_size :  (i+1)*batch_size]
        local_y = y_train[i*batch_size :  (i+1)*batch_size]
        images_batch = torch.from_numpy(local_x).float()
        labels_batch = torch.from_numpy(local_y)

        # Forward pass
        outputs = model(images_batch)
        #criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, torch.max(labels_batch, 1)[1])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


    # evalution after epoch finished
    evalution()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
