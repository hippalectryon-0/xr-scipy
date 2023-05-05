"""Tools for interatcing with docstrings"""

import docstring_parser

SECTIONS = [
    "Args",
    "Arguments",
    "Attributes",
    "Example",
    "Examples",
    "Keyword Args",
    "Keyword Arguments",
    "Note",
    "Notes",
    "Methods",
    "Other Parameters",
    "Parameters",
    "Return",
    "Returns",
    "Raises",
    "References",
    "See Also",
    "See also",
    "Warning",
    "Warnings",
    "Warns",
    "Yield",
    "Yields",
]

ALIASES = {"Return": "Returns", "See also": "See Also"}


class DocParser:
    """A simple parser for sectioning docstrings."""

    def __init__(self, docstring: str):
        assert docstring is not None

        self.parsed_doc = docstring_parser.parse(docstring)

    def insert_description(self, string: str) -> None:
        """insert a description describing the function's new signature"""
        sd, ld = self.parsed_doc.short_description, self.parsed_doc.long_description
        if string.split("(")[0] == sd.split("(")[0]:  # if original doc already has a description, update
            self.parsed_doc.short_description = string
        else:
            self.parsed_doc.short_description = string
            self.parsed_doc.long_description = (
                f"{self.parsed_doc.short_description}\n{self.parsed_doc.long_description}"
                if self.parsed_doc.long_description
                else self.parsed_doc.short_description
            )

    def replace_params(self, **kwargs: docstring_parser.DocstringParam) -> None:
        """replace parameters in the docstring"""
        for i, e in enumerate(self.parsed_doc.params):
            if e.arg_name in kwargs:
                self.parsed_doc.params[i] = kwargs[e.arg_name]

    def remove_params(self, *keys: str) -> None:
        """remove params from docstring"""
        for i, e in enumerate(list(self.parsed_doc.params)):
            if e.arg_name in keys:
                del self.parsed_doc.params[i]

    def remove_sections(self, *keys: str) -> None:
        """remove sections from docstring"""
        for i, e in enumerate(list(self.parsed_doc.meta)):
            if e.args[0] in keys:
                del self.parsed_doc.meta[i]

    def add_params(self, **kwargs: docstring_parser.DocstringParam) -> None:
        """add params to the docstring"""
        self.parsed_doc.params.append(*kwargs)

    def reorder_params(self, *keys: str) -> None:
        """reorder params so that the keys in <args> appear first, in the order provided"""
        new_meta, old_meta = [], list(self.parsed_doc.meta)
        for k in keys:
            new_meta.extend(
                old_meta.pop(i)
                for i, e in enumerate(list(old_meta))
                if isinstance(e, docstring_parser.DocstringParam) and e.arg_name == k
            )
        self.parsed_doc.meta = new_meta + old_meta

    def insert_see_also(self, **kwargs: str) -> None:
        """insert an element in see_also"""
        string = "\n".join(f"{k} : {s}" for k, s in kwargs.items())
        contains_isalso = False
        for e in self.parsed_doc.meta:
            if isinstance(e, docstring_parser.DocstringMeta) and e.args[0] == "see_also":
                e.description += f"\n{string}"
                contains_isalso = True
                continue
        if not contains_isalso:
            self.parsed_doc.meta.append(docstring_parser.DocstringMeta(description=string, args=["see_also"]))

    def __repr__(self) -> str:
        """print this docstrings"""
        return docstring_parser.compose(self.parsed_doc)
