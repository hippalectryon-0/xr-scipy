from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from textwrap import dedent as dedent_


SECTIONS = ['Args', 'Arguments', 'Attributes', 'Example', 'Examples',
            'Keyword Args', 'Keyword Arguments', 'Note', 'Notes', 'Methods',
            'Other Parameters', 'Parameters', 'Return', 'Returns', 'Raises',
            'References', 'See Also', 'See also', 'Warning', 'Warnings',
            'Warns', 'Yield', 'Yields']

ALIASES = {'Return': 'Returns', 'See also': 'See Also'}


def dedent(string):
    """ Similar to textwrap.dedent but neglect the indent of the first
    line. """
    first_line = string.split('\n')[0]
    from_second = dedent_(string[len(first_line)+1:])
    return dedent_(first_line) + '\n' + from_second


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
                key = doc.strip()
                if key in ALIASES:
                    key = ALIASES[key]
                if key not in self.sections:
                    self.sections[key] = []
            else:
                if key is None:
                    self.description.append(doc + '\n')
                else:
                    self.sections[key].append(doc + '\n')
        self.parameters = self._parser_subsection('Parameters')
        self.returns = self._parser_subsection('Returns')
        self.see_also = self._parser_subsection('See Also')

    def _parser_subsection(self, section):
        subsections = OrderedDict()
        if section not in self.sections:
            return subsections

        key = None
        for line in self.sections[section]:
            if len(line) > 0 and line[0] != ' ' and ':' in line:  # title
                key = line.split(':')[0].strip()
                subsections[key] = []
                subsections[key].append(line)
            elif key is not None:
                subsections[key].append(line)

        del self.sections[section]

        return subsections

    def insert_description(self, string):
        if self.description[0] != '\n':
            string = string + '\n'

        funcname = string.strip('(')[0]
        # if original doc already has a description, remove this.
        for i in range(min(2, len(self.description))):
            if funcname + '(' in self.description[i]:
                self.description.pop(i)
                break
        self.description.insert(0, string + '\n')

    def replace_params(self, **kwargs):
        self.parameters = self._replace_subsections(self.parameters, **kwargs)

    def replace_returns(self, **kwargs):
        self.returns = self._replace_subsections(self.returns, **kwargs)

    def _replace_subsections(self, subsection, **kwargs):
        new_subsec = OrderedDict()
        for key, item in subsection.items():
            if key in kwargs.keys():
                new_key = kwargs[key].split(':')[0].strip()
                new_subsec[new_key] = kwargs[key]
            else:
                new_subsec[key] = item
        return new_subsec

    def remove_params(self, *keys):
        for k in keys:
            if k in self.parameters:
                del self.parameters[k]

    def remove_sections(self, *keys):
        for k in keys:
            if k in self.sections:
                del self.sections[k]

    def add_params(self, **kwargs):
        self.parameters.update(kwargs)

    def reorder_params(self, *args):
        params = OrderedDict()
        for k in args:
            if k in self.parameters:
                params[k] = self.parameters.pop(k)
        params.update(self.parameters)
        self.parameters = params

    def insert_see_also(self, **kwargs):
        new_see_also = OrderedDict()
        for k, item in kwargs.items():
            new_see_also[k] = item
        new_see_also.update(self.see_also)
        self.see_also = new_see_also

    def __repr__(self):
        """ print this docstrings """
        docs = ''.join(self.description)
        docs += ''.join(['Parameters\n', '----------\n'])
        for key, item in self.parameters.items():
            docs += ''.join(item)
        if docs[-2] != '\n':
            docs += '\n'

        if len(self.returns) > 0:
            docs += ''.join(['Returns\n', '-------\n'])
            for key, item in self.returns.items():
                docs += ''.join(item)
            if docs[-2] != '\n':
                docs += '\n'

        if len(self.see_also) > 0:
            docs += ''.join(['See Also\n', '--------\n'])
            for key, item in self.see_also.items():
                docs += ''.join(item)
            if docs[-2] != '\n':
                docs += '\n'

        for key, item in self.sections.items():
            docs += key + '\n'
            docs += ''.join(item)

        return docs[:-1]  # remove the last \n
