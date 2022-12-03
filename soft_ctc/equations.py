import numpy as np
from collections import defaultdict


def construct_labeling(confusion_network, blank):
    l = []

    for tau, confusion_set in enumerate(confusion_network):
        l.append((blank, tau))

        for char in confusion_set:
            if char is not None:
                l.append((char, tau))

    l.append((blank, len(confusion_network)))

    return l


def epsilon_probabilities(confusion_network):
    p_epsilon_transition = defaultdict(float)

    for tau, confusion_set in enumerate(confusion_network):
        p_epsilon_transition[tau] = confusion_network[tau][None] if None in confusion_network[tau] else 0

    return p_epsilon_transition


def label_probabilities(confusion_network, labels, epsilon_transition, blank):
    p_label_transition = defaultdict(float)

    for (symbol, tau) in labels:
        if symbol != blank:
            p_label_transition[(symbol, tau)] = confusion_network[tau][symbol]
        else:
            p_label_transition[(symbol, tau)] = 1 - epsilon_transition[tau]

    return p_label_transition


def forward_in(labels, p_symbol):
    p_forward_in = defaultdict(float)

    for (symbol, tau) in labels:
        p_forward_in[(symbol, tau)] = p_symbol[(symbol, tau)]

    return p_forward_in


def forward_out(labels, blank):
    p_forward_out = defaultdict(float)

    for (symbol, tau) in labels:
        if symbol != blank:
            p_forward_out[(symbol, tau)] = 1
        else:
            p_forward_out[(symbol, tau)] = 0

    return p_forward_out


def backward_in(labels, p_symbol, blank):
    p_backward_in = defaultdict(float)

    for (symbol, tau) in labels:
        if symbol != blank:
            p_backward_in[(symbol, tau)] = p_symbol[(symbol, tau)]
        else:
            p_backward_in[(symbol, tau)] = 0

    return p_backward_in


def backward_out(labels):
    p_backward_out = defaultdict(float)

    for (symbol, tau) in labels:
        p_backward_out[(symbol, tau)] = 1

    return p_backward_out


def forward_transition_prob(label1, label2, p_forward_in, p_forward_out, p_epsilon, blank):
    symbol1, tau1 = label1
    symbol2, tau2 = label2

    if tau1 < tau2 and symbol1 != symbol2:
        p = p_forward_out[(symbol1, tau1)] * p_forward_in[(symbol2, tau2)]

        for tau in range(tau1 + 1, tau2):
            p *= p_epsilon[tau]

    elif tau1 == tau2 and symbol1 == blank and symbol2 != blank:
        p = p_forward_in[(symbol2, tau2)] / (1 - p_epsilon[tau2])

    elif tau1 == tau2 and symbol1 == symbol2:
        p = 1

    else:
        p = 0

    return p


def backward_transition_prob(label2, label1, p_backward_in, p_backward_out, p_epsilon, blank):
    symbol2, tau2 = label2
    symbol1, tau1 = label1

    if tau1 < tau2 and symbol1 != symbol2:
        p = p_backward_out[(symbol2, tau2)] * p_backward_in[(symbol1, tau1)]

        for tau in range(tau1 + 1, tau2):
            p *= p_epsilon[tau]

    elif tau1 == tau2 and symbol1 == blank and symbol2 != blank:
        p = p_backward_out[(symbol1, tau1)]

    elif tau1 == tau2 and symbol1 == symbol2:
        p = 1

    else:
        p = 0

    return p


def forward_init(labels, p_forward_in, p_epsilon, dtype=float):
    inits = []

    for (symbol, tau) in labels:
        init_value = p_forward_in[(symbol, tau)]

        for t in range(tau):
            init_value *= p_epsilon[t]

        inits.append(init_value)

    return np.array(inits, dtype=dtype)


def backward_init(labels, p_backward_in, p_backward_out, p_epsilon, blank, dtype=float):
    inits = []

    symbol2, tau2 = labels[-1]

    for (symbol1, tau1) in labels:
        init_value = backward_transition_prob((symbol2, tau2), (symbol1, tau1), p_backward_in, p_backward_out,
                                              p_epsilon, blank)
        inits.append(init_value)

    return np.array(inits, dtype=dtype)
