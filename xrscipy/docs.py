"""Tools for interacting with docstrings"""

from __future__ import annotations

from typing import Any, Callable

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


class CDParam(docstring_parser.DocstringParam):
    """wrapper around DocstringParam"""

    def __init__(self, name: str, description: str, argtype: str = None, optional: bool = False, default: Any = None):
        super().__init__(["param", name], description, name, argtype, optional, default)


class DocParser:
    """A simple parser for sectioning docstrings."""

    def __init__(self, docstring: str = None, fun: Callable = None):
        assert docstring is not None or fun is not None

        self.parsed_doc = docstring_parser.parse(docstring) if docstring else docstring_parser.parse_from_object(fun)

        # Fix snippets
        for e in self.parsed_doc.meta:
            if isinstance(e, docstring_parser.common.DocstringExample):
                e.description = f"{e.snippet}\n{e.description}"

    def insert_description(self, string: str | None) -> None:
        """insert a description describing the function's new signature"""
        if string is None:
            return
        sd, ld = self.parsed_doc.short_description, self.parsed_doc.long_description
        if string.split("(")[0] == sd.split("(")[0]:  # if original doc already has a description, update
            self.parsed_doc.short_description = string
        else:
            self.parsed_doc.short_description = string
            self.parsed_doc.long_description = (
                f"{sd}\n\n{ld}" if self.parsed_doc.long_description else self.parsed_doc.short_description
            )

    def replace_params(self, **kwargs: CDParam) -> None:
        """replace parameters in the docstring"""
        for i, e in enumerate(self.parsed_doc.meta):
            if not isinstance(e, docstring_parser.DocstringParam):
                continue
            if e.arg_name in kwargs:
                self.parsed_doc.meta[i] = kwargs[e.arg_name]

    def remove_params(self, *keys: str) -> None:
        """remove params from docstring"""
        to_remove = []
        for i, e in enumerate(list(self.parsed_doc.meta)):
            if not isinstance(e, docstring_parser.DocstringParam):
                continue
            if e.arg_name in keys:
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            del self.parsed_doc.meta[i]

    def remove_sections(self, *keys: str) -> None:
        """remove sections from docstring"""
        for i, e in enumerate(list(self.parsed_doc.meta)):
            if e.args[0] in keys:
                del self.parsed_doc.meta[i]

    def add_params(self, **kwargs: CDParam) -> None:
        """add params to the docstring"""
        self.parsed_doc.meta.append(*kwargs)

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

    def insert_see_also(self, string: str) -> None:
        """insert an element in see_also"""
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
        return (
            docstring_parser.compose(self.parsed_doc).replace("Seealso\n--------", "See Also\n----------------") + "\n"
        )

    def replace_strings_returns(self, *replacements: tuple[str, str]) -> None:
        """replaces strings in returns"""
        for e in self.parsed_doc.meta:
            if not isinstance(e, docstring_parser.DocstringReturns):
                continue
            for rep_from, rep_to in replacements:
                if e.description:
                    e.description.replace(rep_from, rep_to)

    def get_parameter(self, name: str) -> docstring_parser.DocstringParam | None:
        """get parameter from name"""
        return next((e for e in self.parsed_doc.params if e.arg_name == name), None)

    def replace_strings_description(self, *replacements: tuple[str, str]) -> None:
        """replaces strings in description"""
        for rep_from, rep_to in replacements:
            self.parsed_doc.long_description = self.parsed_doc.long_description.replace(rep_from, rep_to)
            self.parsed_doc.short_description = self.parsed_doc.short_description.replace(rep_from, rep_to)
