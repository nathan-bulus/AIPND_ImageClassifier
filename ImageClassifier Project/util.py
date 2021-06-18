import torch
from torchvision import datasets, transforms, models


def load_data(data_dir):

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms ={
        'training' : transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])
                                    ]),
        'validation' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ]),
        'testing' : transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])
    }

    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    dataloaders = {
        'training' :torch.utils.data.DataLoader(image_datasets['training'], batch_size=64,                  shuffle=True),
        'validation' :torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64),
        'testing' :torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64)
    }



    train_data = image_datasets['training']
    valid_data = image_datasets['validation']
    test_data = image_datasets['testing']

    train_loader = dataloaders['training']
    valid_loader = dataloaders['validation']
    test_loader = dataloaders['testing']

    return train_loader, valid_loader, test_loader, train_data, test_data, valid_data



def train_model(model, epochs, train_loader, valid_loader, criterion, optimizer, gpu_mode):
    steps = 0
    print_every = 50

    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    for epoch in range(epochs):
        training_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps +=1

            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for jj, (inputs, labels) in enumerate(valid_loader):
                        if gpu_mode == True:
                            inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        else:
                            pass
                        outputs = model.forward(inputs)
                        loss =criterion(outputs, labels)

                        valid_loss += loss.item()
                        ps = torch.exp(outputs).data
                        equals = (labels.data == ps.max(1)[1])
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch: {epoch+1}/{epochs}... "
                         f"Training Loss: {training_loss/len(train_loader):.3f}.. "
                         f"validation Loss: {valid_loss:.3f}.. "
                         f"Validation Accuracy %: {100 * accuracy/len(valid_loader)}%")
                    training_loss = 0

    return model, optimizer



def test_model(model, test_loader, gpu_mode):
    test_accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(test_loader):

            if gpu_mode == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                pass
            # inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_accuracy += (pred == labels).sum().item()

    print(f"Accuracy Of Neural Network: {round(100 * test_accuracy / total,2)}%")
