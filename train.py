import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    # Model
    model = torch.nn.Sequential(
        torch.nn.Linear(3 * 32 * 32, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train
    for epoch in range(5):  # 5 эпох для простоты
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs = inputs.view(-1, 3 * 32 * 32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save
    torch.save(model.state_dict(), "model.pth")
