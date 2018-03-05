from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from textwrap import dedent


SECTIONS = ['Args', 'Arguments', 'Attributes', 'Example', 'Examples',
            'Keyword Args', 'Keyword Arguments', 'Note', 'Notes',
            'Other Parameters', 'Parameters', 'Return', 'Returns', 'Raises',
            'References', 'See Also', 'Warning', 'Warnings', 'Warns',
            'Yield', 'Yields']


class DocParser(object):
    def __init__(self, docstring):
        """ A simple parser for sectioning docstrings. """
        docstring = dedent(docstring)
        self.description = []
        key = None
        self.sections = OrderedDict()

        # parse
        docstring = docstring.split('\n')
        for doc in docstring:
            if doc.strip() in SECTIONS:
                key = doc
                self.sections[key] = []
            else:
                if key is None:
                    self.description.append(doc + '\n')
                else:
                    self.sections[key].append(doc + '\n')
        self._parse_parameters()

    def _parse_parameters(self):
        """ parse self.sections['Parameters'] """
        self.parameters = OrderedDict()
        if 'Parameters' not in self.sections:
            return

        key = None
        for line in self.sections['Parameters']:
            if len(line) > 0 and line[0] != ' ' and ':' in line:  # title
                key = line.split(':')[0].strip()
                self.parameters[key] = []
                self.parameters[key].append(line)
            elif key is not None:
                self.parameters[key].append(line)

        del self.sections['Parameters']

    def replace_param(self, new_key, new_doc):
        parameters = OrderedDict()
        for key, item in self.parameters.items():
            if key == new_key:
                parameters[key] = new_doc
            else:
                parameters[key] = item
        self.parameters = parameters

    def __repr__(self):
        """ print this docstrings """
        docs = ''.join(self.description)
        docs += ''.join(['Parameters\n', '----------\n'])
        for key, item in self.parameters.items():
            docs += ''.join(item)
        for key, item in self.sections.items():
            docs += key + '\n'
            docs += ''.join(item)

        return docs[:-1]  # remove the last \n
