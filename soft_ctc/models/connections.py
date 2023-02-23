import numpy as np
from scipy import sparse

from soft_ctc import equations as eqs


class Connections:
    def __init__(self, forward, backward, forward_start, forward_end, backward_start, backward_end, is_sparse=False):
        self.forward = forward
        self.backward = backward
        self.forward_start = forward_start
        self.forward_end = forward_end
        self.backward_start = backward_start
        self.backward_end = backward_end

        self._sparse = is_sparse

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
        is_sparse = self._sparse
        if is_sparse:
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

        if is_sparse:
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
    def from_confusion_network(confusion_network, blank=0, labeling=None, dtype=np.float):
        if labeling is None:
            labeling = eqs.construct_labeling(confusion_network, blank)

        label_probs = [eqs.p_symbol(confusion_network, symbol, tau, blank) if tau < len(confusion_network) else 1.0
                       for (symbol, tau) in labeling]
        label_probs = np.array(label_probs, dtype=dtype)

        forward_matrix = np.full((len(labeling), len(labeling)), -1.0, dtype=dtype)

        for i, (symbol1, tau1) in enumerate(labeling):
            for j, (symbol2, tau2) in enumerate(labeling):
                forward_matrix[i, j] = eqs.p_transition(confusion_network, symbol1, tau1, symbol2, tau2, blank)

        backward_matrix = np.transpose(forward_matrix)

        forward_init_vector = eqs.alpha_init(confusion_network, labeling, blank, dtype=dtype)
        backward_init_vector = eqs.beta_init(confusion_network, labeling, blank, dtype=dtype)
        forward_loss_vector = backward_init_vector
        backward_loss_vector = forward_init_vector / label_probs

        return Connections(forward_matrix, backward_matrix,
                           forward_init_vector, forward_loss_vector,
                           backward_init_vector, backward_loss_vector)


def convert_characters_to_labels(confusion_network, character_set):
    return [{character_set.index(char) if char is not None else None: prob for char, prob in confusion_set.items()}
            for confusion_set in confusion_network]


def convert_confusion_network_to_connections(confusion_network, blank, dtype=float):
    labeling = eqs.construct_labeling(confusion_network, blank)
    labels = [symbol for (symbol, tau) in labeling]

    connections = Connections.from_confusion_network(confusion_network, blank=blank, labeling=labeling, dtype=dtype)

    return labels, connections


def main():
    BLANK = "<BLANK>"

    character_set = [BLANK] + list("ACESTU")
    blank = character_set.index(BLANK)

    confusion_network = [{'C': 0.5, None: 0.5}, {'A': 0.7, 'U': 0.2, None: 0.1}, {'T': 1.0}, {'E': 0.6, 'S': 0.4}]
    confusion_network = convert_characters_to_labels(confusion_network, character_set)

    labels, connections = convert_confusion_network_to_connections(confusion_network, blank)

    np.set_printoptions(precision=3, floatmode="fixed")

    print("Confusion network")
    print(confusion_network)
    print()

    print("Labels")
    print(labels)
    print()

    print("Connections")
    print(connections)

    return 0


if __name__ == "__main__":
    exit(main())
