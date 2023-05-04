"""test docs"""
from textwrap import dedent

from xrscipy import docs


def example_func(_a, _b):
    """
    An example of function.

    This is just an example.

    Parameters
    ----------
    _a : int
        An example argument.
    _b : float
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
    assert repr(parser) == dedent(example_func.__doc__)

    parser.replace_params(_a='_c : int\n    Replaced parameter.\n')
