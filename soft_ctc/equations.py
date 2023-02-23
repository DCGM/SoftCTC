import numpy as np


def construct_labeling(confusion_network, blank):
    l = []

    for tau, confusion_set in enumerate(confusion_network):
        l.append((blank, tau))

        for char in confusion_set:
            if char is not None:
                l.append((char, tau))

    l.append((blank, len(confusion_network)))

    return l


def p_epsilon(confusion_network, tau):
    if None in confusion_network[tau]:
        p = confusion_network[tau][None]
    else:
        p = 0.0

    return p


def p_symbol(confusion_network, symbol, tau, blank):
    if symbol == blank:
        p = 1.0 - p_epsilon(confusion_network, tau)
    else:
        p = confusion_network[tau][symbol]

    return p


def p_in(confusion_network, symbol, tau, blank):
    return p_symbol(confusion_network, symbol, tau, blank)


def p_out(symbol, blank):
    if symbol == blank:
        p = 0.0
    else:
        p = 1.0

    return p


def p_transition(confusion_network, symbol1, tau1, symbol2, tau2, blank):
    if tau1 < tau2 and symbol1 != symbol2:
        p_out_symbol1 = p_out(symbol1, blank)
        p_in_symbol2 = p_in(confusion_network, symbol2, tau2, blank) if tau2 < len(confusion_network) else 1.0

        p = p_out_symbol1 * p_in_symbol2

        for tau in range(tau1 + 1, tau2):
            p *= p_epsilon(confusion_network, tau)

    elif tau1 == tau2 and symbol1 == blank and symbol2 != blank:
        p = p_in(confusion_network, symbol2, tau2, blank) / (1.0 - p_epsilon(confusion_network, tau2))

    elif symbol1 == symbol2 and tau1 == tau2:
        p = 1.0

    else:
        p = 0.0

    return p


def alpha_init(confusion_network, labels, blank, dtype=np.float):
    inits = []

    for symbol, tau in labels:
        if tau < len(confusion_network):
            init_value = p_in(confusion_network, symbol, tau, blank)
        else:
            init_value = 1.0

        for t in range(tau):
            init_value *= p_epsilon(confusion_network, t)

        inits.append(init_value)

    return np.array(inits, dtype=dtype)


def beta_init(confusion_network, labels, blank, dtype=np.float):
    inits = []

    for symbol, tau in labels:
        if tau < len(confusion_network):
            init_value = p_out(symbol, blank)
        else:
            init_value = 1.0

        for t in range(tau+1, len(confusion_network)):
            init_value *= p_epsilon(confusion_network, t)

        inits.append(init_value)

    return np.array(inits, dtype=dtype)
