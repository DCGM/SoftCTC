import numpy as np
from scipy import sparse

from soft_ctc import equations as eqs


class Connections:
    def __init__(self, forward, backward, forward_start, forward_end, backward_start, backward_end, sparse=False):
        self.forward = forward
        self.backward = backward
        self.forward_start = forward_start
        self.forward_end = forward_end
        self.backward_start = backward_start
        self.backward_end = backward_end

        self._sparse = sparse

    def size(self):
        size = self.forward_start.shape[0]

        if self._sparse:
            size = self.forward_start.shape[1]

        return size

    def is_sparse(self):
        return self._sparse

    def _dense_to_sparse(self, matrix):
        return sparse.csr_matrix(matrix)

    def _sparse_to_dense(self, matrix):
        return matrix.toarray()

    def to_sparse(self):
        if not self._sparse:
            self.forward = self._dense_to_sparse(self.forward)
            self.backward = self._dense_to_sparse(self.backward)
            self.forward_start = self._dense_to_sparse(self.forward_start)
            self.forward_end = self._dense_to_sparse(self.forward_end)
            self.backward_start = self._dense_to_sparse(self.backward_start)
            self.backward_end = self._dense_to_sparse(self.backward_end)
            self._sparse = True

    def to_dense(self):
        if self._sparse:
            self.forward = self._sparse_to_dense(self.forward)
            self.backward = self._sparse_to_dense(self.backward)

            # The .reshape(-1) transforms the shape of the vector from (1, N) to (N,). SciPy does not keep the original
            # vector shape.
            self.forward_start = self._sparse_to_dense(self.forward_start).reshape(-1)
            self.forward_end = self._sparse_to_dense(self.forward_end).reshape(-1)
            self.backward_start = self._sparse_to_dense(self.backward_start).reshape(-1)
            self.backward_end = self._sparse_to_dense(self.backward_end).reshape(-1)
            self._sparse = False

    def __str__(self):
        output = "Forward transitions\n"
        output += str(self.forward)
        output += "\n\n"

        output += "Backward transitions\n"
        output += str(self.backward)
        output += "\n\n"

        output += "Forward init vector\n"
        output += str(self.forward_start)
        output += "\n\n"

        output += "Forward loss vector\n"
        output += str(self.forward_end)
        output += "\n\n"

        output += "Backward init vector\n"
        output += str(self.backward_start)
        output += "\n\n"

        output += "Backward loss vector\n"
        output += str(self.backward_end)

        return output

    def extend(self, target_size):
        sparse = self._sparse
        if sparse:
            self.to_dense()

        current_size = self.size()

        if target_size == current_size:
            extended_connections = Connections(np.copy(self.forward), np.copy(self.backward),
                                               np.copy(self.forward_start), np.copy(self.forward_end),
                                               np.copy(self.backward_start), np.copy(self.backward_end))

        elif target_size > current_size:
            forward = Connections._extend_2d(self.forward, target_size)
            forward_start = Connections._extend_1d(self.forward_start, target_size)
            forward_end = Connections._extend_1d(self.forward_end, target_size)

            backward = Connections._extend_2d(self.backward, target_size)
            backward_start = Connections._extend_1d(self.backward_start, target_size)
            backward_end = Connections._extend_1d(self.backward_end, target_size)

            extended_connections = Connections(forward, backward,
                                               forward_start, forward_end,
                                               backward_start, backward_end)
        else:
            extended_connections = None

        if sparse:
            self.to_sparse()

            if extended_connections is not None:
                extended_connections.to_sparse()

        return extended_connections

    @staticmethod
    def _extend_2d(matrix, target_size):
        height, width = matrix.shape
        extended_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
        extended_matrix[:height, :width] = matrix

        return extended_matrix

    @staticmethod
    def _extend_1d(vector, target_size):
        length = vector.shape[0]
        extended_vector = np.zeros(target_size, dtype=vector.dtype)
        extended_vector[:length] = vector

        return extended_vector

    @staticmethod
    def from_confusion_network(confusion_network, blank=0, labeling=None, epsilon_probs=None, label_probs=None,
                               dtype=float):
        if labeling is None:
            labeling = eqs.construct_labeling(confusion_network, blank)

        if epsilon_probs is None:
            epsilon_probs = eqs.epsilon_probabilities(confusion_network)

        if label_probs is None:
            label_probs = eqs.label_probabilities(confusion_network, labeling, epsilon_probs, blank)

        p_forward_in = eqs.forward_in(labeling, label_probs)
        p_forward_out = eqs.forward_out(labeling, blank)
        p_backward_in = eqs.backward_in(labeling, label_probs, blank)
        p_backward_out = eqs.backward_out(labeling)

        forward_init_vector = eqs.forward_init(labeling, p_forward_in, epsilon_probs)
        backward_init_vector = eqs.backward_init(labeling, p_backward_in, p_backward_out, epsilon_probs, blank)

        forward_transitions = np.full((len(labeling), len(labeling)), -1, dtype=dtype)
        backward_transitions = np.copy(forward_transitions)

        for index1, label1 in enumerate(labeling):
            for index2, label2 in enumerate(labeling):
                forward_transitions[index1, index2] = eqs.forward_transition_prob(label1, label2, p_forward_in,
                                                                                  p_forward_out, epsilon_probs, blank)
                backward_transitions[index1, index2] = eqs.backward_transition_prob(label1, label2, p_backward_in,
                                                                                    p_backward_out, epsilon_probs, blank)

        label_probs_vector = np.array([label_probs[label] for label in labeling])
        forward_loss_vector = backward_init_vector / label_probs_vector
        backward_loss_vector = forward_init_vector / label_probs_vector

        return Connections(forward_transitions, backward_transitions,
                           forward_init_vector, forward_loss_vector,
                           backward_init_vector, backward_loss_vector)


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


def convert_confusion_network_to_connections(confusion_network, blank):
    labeling = eqs.construct_labeling(confusion_network, blank)
    epsilon_probs = eqs.epsilon_probabilities(confusion_network)

    labels = [symbol for (symbol, tau) in labeling]

    label_probs_dict = eqs.label_probabilities(confusion_network, labeling, epsilon_probs, blank)
    label_probs = [label_probs_dict[label] for label in labeling]

    connections = Connections.from_confusion_network(confusion_network, blank=blank, labeling=labeling,
                                                     epsilon_probs=epsilon_probs, label_probs=label_probs_dict)

    return labels, label_probs, connections


def main():
    BLANK = "<BLANK>"

    character_set = [BLANK] + list("ACESTU")
    blank = character_set.index(BLANK)

    confusion_network = [{'C': 0.5, None: 0.5}, {'A': 0.7, 'U': 0.2, None: 0.1}, {'T': 1.0}, {'E': 0.6, 'S': 0.4}]
    confusion_network = convert_characters_to_labels(confusion_network, character_set)

    labels, label_probs, connections = convert_confusion_network_to_connections(confusion_network, blank)
    label_probs = np.array(label_probs)

    np.set_printoptions(precision=3, floatmode="fixed")

    print("Confusion network")
    print(confusion_network)
    print()

    print("Labels")
    print(labels)
    print()

    print("Label probabilities")
    print(label_probs)
    print()

    print("Connections")
    print(connections)

    return 0


if __name__ == "__main__":
    exit(main())
