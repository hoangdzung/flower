"""Util functions for MedMNIST."""
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History
from PIL import Image
from torch import Tensor, load
from torch.nn import GroupNorm, Module
from torch.utils.data import DataLoader, Dataset
from .custom_resnet import Model
from .data_utils import get_dataclass
from medmnist import *
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr_baselines.dataset.utils.common import (
    XY,
    create_lda_partitions,
    shuffle,
    sort_by_label,
    split_array_at_indices,
)

# transforms
def get_transforms() -> Dict[str, Compose]:
    """Returns the right Transform Compose for both train and evaluation.

    Returns:
        Dict[str, Compose]: Dictionary with 'train' and 'test' keywords and Transforms
        for each
    """
    transform=Compose([ToTensor(), Normalize(mean=[.5], std=[.5])])
    return {"train": transform, "test": transform}


def get_model(num_classes: int = 10, num_channels: int = 1) -> Module:
    """Generates ResNet18 model using GroupNormalization rather than
    BatchNormalization. Two groups are used.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        num_channels (int, optional): Number of input channels. Defaults to 3.

    Returns:
        Module: ResNet12 network.
    """
    model: Model = Model(num_classes=num_classes, in_channels=num_channels)
    return model


class ClientDataset(Dataset):
    """Client Dataset."""

    def __init__(self, path_to_data: Path, transform: Compose = None):
        """Implements local dataset.

        Args:
            path_to_data (Path): Path to local '.pt' file is located.
            transform (Compose, optional): Transforms to be used when sampling.
            Defaults to None.
        """
        super().__init__()
        self.transform = transform
        self.inputs, self.labels = load(path_to_data)

    def __len__(self) -> int:
        """Size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Fetches item in dataset.

        Args:
            idx (int): Position of item being fetched.

        Returns:
            Tuple[Tensor, int]: Tensor image and respective label
        """
        this_input = Image.fromarray(self.inputs[idx])
        this_label = self.labels[idx]
        if self.transform:
            this_input = self.transform(this_input)

        return this_input, this_label


def save_partitions(
    list_partitions: List[XY], fed_dir: Path, partition_type: str = "train"
):
    """Saves partitions to individual files.

    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / f"{partition_type}.pt")


def partition_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: Optional[npt.NDArray[np.float32]] = None,
    num_partitions: int = 500,
    concentration: float = 0.1,
) -> np.ndarray:
    """Creates and saves partitions

    Args:
        dataset (XY): Original complete dataset.
        fed_dir (Path): Root directory where to save partitions.
        dirichlet_dist (Optional[npt.NDArray[np.float32]], optional):
            Pre-defined distributions to be used for sampling if exist. Defaults to None.
        num_partitions (int, optional): Number of partitions. Defaults to 500.
        concentration (float, optional): Alpha value for Dirichlet. Defaults to 0.1.

    Returns:
        np.ndarray: Generated dirichlet distributions.
    """
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
        accept_imbalanced=True
    )
    # Save partions
    save_partitions(list_partitions=clients_partitions, fed_dir=fed_dir)

    return dist

    
def get_dataset(path_original_dataset: Path, dataset_name: str, train: bool = True, transform = None):
    dataclass = get_dataclass(dataset_name)
    return dataclass(root=path_original_dataset, split='train' if train else 'test', download=True, transform = transform)

def gen_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
    lda_concentration: float,
) -> Path:
    """Defines root path for partitions and calls functions to create them.

    Args:
        path_original_dataset (Path): Path to original (unpartitioned) dataset.
        dataset_name (str): Friendly name to dataset.
        num_total_clients (int): Number of clients.
        lda_concentration (float): Concentration (alpha) used when generation Dirichlet
        distributions.

    Returns:
        Path: [description]
    """
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
        / f"{lda_concentration:.2f}"
    )

#     trainset = CIFAR10(root=path_original_dataset, train=True, download=True)
    trainset = get_dataset(path_original_dataset, dataset_name, train=True)
    flwr_trainset = (trainset.imgs, np.array(trainset.labels, dtype=np.int32).squeeze())
    partition_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
        concentration=lda_concentration,
    )

    return fed_dir

def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device).long().squeeze()
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": server_round,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def get_eval_fn(
    path_original_dataset: Path, dataset_name: str, num_classes: int = 10, num_channels: int = 10
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Returns an evaluation function for centralized evaluation."""
    transforms = get_transforms()
    
    # need to change it
#     testset = CIFAR(
#         root=path_original_dataset,
#         train=False,
#         download=True,
#         transform=transforms["test"],
#     )
    testset = get_dataset(path_original_dataset, dataset_name, train=False, transform=transforms["test"])
    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire test set for evaluation."""
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_model(num_classes=num_classes, num_channels=num_channels)
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def get_initial_parameters(num_classes: int = 10, num_channels: int = 3) -> Parameters:
    """Returns initial parameters from a model.

    Args:
        num_classes (int, optional)
        num_channels (int, optional)

    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    model = get_model(num_classes, num_channels)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
) -> None:
    """Simple plotting method for Classification Task.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.close()
