from __future__ import absolute_import, division, print_function

from textwrap import dedent

from xrscipy import docs


def example_func(a, b):
    """
    An example of function.

    This is just an example.

    Parameters
    ----------
    a : int
        An example argument.
    b : float
        Another example argument

    See Also
    --------
    xrscipy.docs: original scipy implementation

    Note
    ----
    This is a note
    """
    pass


def test_doc_parser():
    parser = docs.DocParser(example_func.__doc__)
    print(parser)
    assert repr(parser) == dedent(example_func.__doc__)

    parser.replace_params(a="c : int\n    Replaced parameter.\n")
