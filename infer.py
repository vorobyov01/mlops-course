import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    # Load model
    model = torch.nn.Sequential(
        torch.nn.Linear(3 * 32 * 32, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 2),
    )
    model.load_state_dict(torch.load("model.pth"))

    # Load val data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 3 * 32 * 32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")

    # Save prediction
    df = pd.DataFrame(predictions, columns=["Prediction"])
    df.to_csv("predictions.csv", index=False)
