import os
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

# 1. LoadData
data_dir = "./Dogs&Cats/"
data_transform = {x:transforms.Compose([transforms.Resize((64,64),interpolation=3),
                                        transforms.ToTensor()])
                  for x in ["train", "valid"]}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                            batch_size=16,
                                            shuffle=True)
              for x in ["train", "valid"]}

# get a batch data
x_example, y_example = next(iter(dataloader["train"]))
print("x example number:",len(x_example)) # batch_size, C , H, W
print("y example number:",len(y_example))

index_classes = image_datasets["train"].class_to_idx
# print(index_classes)
example_classes = image_datasets["train"].classes
# print(example_classes)

'''
img = torchvision.utils.make_grid(x_example)
img = img.numpy().transpose([1,2,0])
print([example_classes[i] for i in y_example])
plt.imshow(img)
plt.show()
'''

'''
# 2. Model
# VGG
class Models(torch.nn.Module):

    def __init__(self):
        super(Models, self).__init__()  # 执行父类的构造函数

        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # 32 x 32

            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # 16 x 16

            torch.nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # 8 x 8

            torch.nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),  # 4 x 4
            )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(4*4*512,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,2)
            )

    def forward(self, input):
        x = self.Conv(input)
        x = x.view(-1,4*4*512)
        x = self.Classes(x)

        return x

model = Models()
print(model)
'''

# 2*. transfer RESNET50
model = models.resnet50(pretrained=True)

for para in model.parameters():
    para.requires_grad = False

model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
model.fc = torch.nn.Linear(2048,2)
print(model)

# 3. Training
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


use_gpu = torch.cuda.is_available()
print("CUDA is availabel:", use_gpu)
if use_gpu:
    model = model.cuda()

epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*10)

    for phase in ["train","valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)

        # including train and valid
        running_loss = 0.0
        running_correct = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            y_pred = model(X)
            loss = loss_f(y_pred, y)

            optimizer.zero_grad()
            if phase=="train":
                loss.backward()
                optimizer.step()

            _,pred = torch.max(y_pred.data, 1)
            running_loss += loss.data
            running_correct += torch.sum(pred==y.data)

            if batch%200==00 and phase=="train":
                print("Batch {}, Training Loss:{:.4f}, Train Accuracy:{:.4f}%".format(
                    batch, running_loss/batch, 100*running_correct/(16*batch)))

        epoch_loss = 16*running_loss / len(image_datasets[phase])
        epoch_acc = 100*running_correct / len(image_datasets[phase])

        print("{} Loss:{:.4f} Accuracy:{:.4f}%".format(phase, epoch_loss, epoch_acc))

    time_end = time.time() - time_open
    print(time_end)

	


