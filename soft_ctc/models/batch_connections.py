import torch
import numpy as np
from typing import Optional, List, Dict

from soft_ctc import equations as eqs
from soft_ctc.models.connections import Connections, convert_confusion_network_to_connections


class BatchConnections():
    def __init__(self, forward, backward, forward_start, forward_end, backward_start, backward_end):
        self.forward = forward
        self.backward = backward
        self.forward_start = forward_start
        self.forward_end = forward_end
        self.backward_start = backward_start
        self.backward_end = backward_end

    def to(self, device):
        self.forward = self.forward.to(device)
        self.forward_start = self.forward_start.to(device)
        self.forward_end = self.forward_end.to(device)
        self.backward = self.backward.to(device)
        self.backward_start = self.backward_start.to(device)
        self.backward_end = self.backward_end.to(device)
        return self

    def torch(self):
        self.forward = torch.from_numpy(self.forward)
        self.forward_start = torch.from_numpy(self.forward_start)
        self.forward_end = torch.from_numpy(self.forward_end)
        self.backward = torch.from_numpy(self.backward)
        self.backward_start = torch.from_numpy(self.backward_start)
        self.backward_end = torch.from_numpy(self.backward_end)
        return self

    def numpy(self):
        self.forward = self.forward.numpy()
        self.forward_start = self.forward_start.numpy()
        self.forward_end = self.forward_end.numpy()
        self.backward = self.backward.numpy()
        self.backward_start = self.backward_start.numpy()
        self.backward_end = self.backward_end.numpy()
        return self

    def device(self):
        return self.forward_start.device

    def size(self):
        return self.forward_start.shape[1]

    def __len__(self):
        return self.forward_start.shape[0]

    def __getitem__(self, index):
        return Connections(self.forward[index], self.backward[index],
                           self.forward_start[index], self.forward_end[index],
                           self.backward_start[index], self.backward_end[index])

    def __str__(self):
        output = f"Forward transitions: {self.forward.shape}\n"
        output += f"Forward init vector: {self.forward_start.shape}\n"
        output += f"Forward loss vector: {self.forward_end.shape}\n"
        output += f"Backward transitions: {self.backward.shape}\n"
        output += f"Backward init vector: {self.backward_start.shape}\n"
        output += f"Backward loss vector: {self.backward_end.shape}"
        return output

    @staticmethod
    def stack_connections(connections: List[Connections], target_connections_size: Optional[int] = None):
        sparse_connections = []
        for c in connections:
            if c.is_sparse():
                c.to_dense()
                sparse_connections.append(True)
            else:
                sparse_connections.append(False)

        connections_sizes = [c.size() for c in connections]
        if target_connections_size is not None:
            connections_sizes += [target_connections_size]

        target_connections_size = max(connections_sizes)
        extended_connections = [c.extend(target_connections_size) for c in connections]

        for sparse, c in zip(sparse_connections, connections):
            if sparse:
                c.to_sparse()

        forward = np.stack([c.forward for c in extended_connections])
        forward_start = np.stack([c.forward_start for c in extended_connections])
        forward_end = np.stack([c.forward_end for c in extended_connections])
        backward = np.stack([c.backward for c in extended_connections])
        backward_start = np.stack([c.backward_start for c in extended_connections])
        backward_end = np.stack([c.backward_end for c in extended_connections])

        return BatchConnections(forward, backward, forward_start, forward_end, backward_start, backward_end)

    @staticmethod
    def from_confusion_networks(confusion_networks: List[Dict[Optional[str], float]],
                                target_connections_size: Optional[int] = None,
                                dtype=float):
        connections = [Connections.from_confusion_network(cn, dtype=dtype) for cn in confusion_networks]
        return BatchConnections.stack_connections(connections, target_connections_size)


def stack_labels_and_label_probs(labels, label_probs, blank, target_size=None):
    if target_size is None:
        target_size = max([len(l) for l in labels])

    for index, (_labels, _label_probs) in enumerate(zip(labels, label_probs)):
        labels_padding = [blank] * (target_size - len(_labels))
        labels[index] += labels_padding
        labels[index] = np.array(labels[index])

        label_probs_padding = [1.0] * (target_size - len(_label_probs))
        label_probs[index] += label_probs_padding
        label_probs[index] = np.array(label_probs[index])

    labels = np.stack(labels)
    label_probs = np.stack(label_probs)

    return labels, label_probs


def calculate_target_size(sizes: List[int], size_coefficient=64):
    return int(np.ceil(np.max(sizes) / size_coefficient) * size_coefficient)


def convert_characters_to_labels(confusion_network, character_set):
    new_confusion_network = []

    for confusion_set in confusion_network:
        new_confusion_set = {}
        for character in confusion_set:
            if character is None:
                new_confusion_set[None] = confusion_set[None]
            else:
                new_confusion_set[character_set.index(character)] = confusion_set[character]

        new_confusion_network.append(new_confusion_set)

    return new_confusion_network


def main():
    BLANK = "<BLANK>"

    character_set = [BLANK] + list("ACESTU")
    blank = character_set.index(BLANK)

    confusion_network = [{'C': 0.5, None: 0.5}, {'A': 0.7, 'U': 0.2, None: 0.1}, {'T': 1.0}, {'E': 0.6, 'S': 0.4}]
    confusion_network = convert_characters_to_labels(confusion_network, character_set)

    labels, label_probs, connections = convert_confusion_network_to_connections(confusion_network, blank)

    N = 4
    SIZE_COEFFICIENT = 8
    
    target_size = calculate_target_size(connections.size(), size_coefficient=SIZE_COEFFICIENT)
    batch_connections = BatchConnections.stack_connections([connections for _ in range(N)], target_size)
    batch_labels, batch_label_probs = stack_labels_and_label_probs([labels for _ in range(N)],
                                                                   [label_probs for _ in range(N)], blank,
                                                                   target_size)
    np.set_printoptions(precision=3, floatmode="fixed")

    print("Target size")
    print(target_size)
    print()

    print("Batch labels")
    print(batch_labels)
    print()

    print("Batch label probabilities")
    print(batch_label_probs)
    print()

    print("Batch connections")
    print(str(batch_connections))

    return 0


if __name__ == "__main__":
    exit(main())
