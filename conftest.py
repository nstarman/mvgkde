"""Doctest configuration."""

from collections.abc import Callable, Iterable, Sequence
from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Document, Region, Sybil
from sybil.parsers.myst import (
    DocTestDirectiveParser as MarkdownDocTestDirectiveParser,
    PythonCodeBlockParser as MarkdownPythonCodeBlockParser,
    SkipParser as MarkdownSkipParser,
)
from sybil.parsers.rest import DocTestParser as ReSTDocTestParser

optionflags = ELLIPSIS | NORMALIZE_WHITESPACE

parsers: Sequence[Callable[[Document], Iterable[Region]]] = [
    MarkdownDocTestDirectiveParser(optionflags=optionflags),
    MarkdownPythonCodeBlockParser(doctest_optionflags=optionflags),
    MarkdownSkipParser(),
]

docs = Sybil(parsers=parsers, patterns=["*.md"])
python = Sybil(
    parsers=[ReSTDocTestParser(optionflags=optionflags), *parsers],
    patterns=["*.py"],
)


pytest_collect_file = (docs + python).pytest()
