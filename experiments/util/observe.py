from chainer import functions as F


def observe_until_click(c):
    """
    Generates a vector of observed documents (=1) and unobserved documents (=0)
    under the assumption that the user observed all documents up to and
    including the last clicked one.

    :param c: The click vectors
    :type c: chainer.Variable

    :return: The observation vectors
    :rtype: chainer.Variable
    """
    return F.fliplr(F.clip(F.cumsum(F.fliplr(c), axis=1), 0.0, 1.0))

