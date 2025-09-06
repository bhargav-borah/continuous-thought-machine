from torchvision import datasets, transforms
import torch

def get_mnist(train_transform=None,
              test_transform=None,
              batch_size=512,
              num_workers=2,
              pin_memory=True
              ):
  """
  Prepare train and test dataloaders

  Arguments:
    train_transform: Transforms to apply to the train data (default None)
    test_transform: Transforms to apply to the test data (default None)
    batch_size: Batch size for train and test dataloaders (default 64)
    num_workers: Number of separate subprocesses which load batches in parallel (default -1)
    pin_memory: If True, will copy tensors into pinned (page-locked), enabling faster transfer of batches to GPU(s); set to False if not using GPU (Default True)

  Returns:
    trainloader: A torch.utils.data.DataLoader object which yields train data in batches
    testloader: A torch.utils.data.DataLoader object which yields test data in batches
  """
  if train_transform is None:
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

  if test_transform is None:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

  train_data = datasets.MNIST(
      root='./data',
      train=True,
      download=True,
      transform=train_transform
  )
  test_data = datasets.MNIST(
      root='./data',
      train=True,
      download=True,
      transform=test_transform
  )

  if not torch.cuda.is_available():
    pin_memory = False
    
  trainloader = torch.utils.data.DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=pin_memory
  )
  testloader = torch.utils.data.DataLoader(
      dataset=test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=pin_memory
  )

  return trainloader, testloader