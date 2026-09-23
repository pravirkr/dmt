from __future__ import annotations

import datetime
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

# Add project source only when the installed package is not importable.
# Prepending src/ unconditionally shadows the wheel on Read the Docs, where
# the compiled extension lives in site-packages rather than the checkout.
DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Prefer the checkout when its compiled extension is sitting next to the
# sources. Otherwise (Read the Docs, plain pip install) use the installed
# package, whose extension is not in the source tree.
if any(SRC_DIR.glob("dmtlib/libdmt*.so")):
    sys.path.insert(0, str(SRC_DIR))

try:
    import dmtlib
except ImportError:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        import dmtlib
    except ImportError:
        dmtlib = None  # type: ignore[assignment]

# -- Project information -----------------------------------------------------
project = "dmt"
author = "Pravir Kumar"
year = datetime.datetime.now(tz=datetime.UTC).date().year
copyright = f"{year}, {author}"  # noqa: A001
release = getattr(dmtlib, "__version__", None) or "0.2.0"
version = release
master_doc = "index"
repo_url = "https://github.com/pravirkr/dmt"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

# Breathe is registered only when Doxygen XML is actually present.
# A missing or empty index used to abort configuration.

templates_path = []
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "Doxyfile",
    "Doxyfile.in",
    "CMakeLists.txt",
    # Logo generator and its exports, not documentation pages.
    "logo-src",
    # Written by conf.py and included from api/cpp_api.md. Not a standalone page.
    "api/_breathe_body.rst",
]
# "any" turns ordinary RST backticks into cross-references and fails the
# build whenever a name is not in the domain index. Leave the default role
# unset so backticks stay literal code.
nitpicky = False
rst_epilog = f"""
.. |project| replace:: {project}
"""

# -- HTML
html_theme = "sphinx_book_theme"
html_context = {"default_mode": "light"}
html_title = project
html_last_updated_fmt = "%b %d, %Y"
html_theme_options = {
    "repository_url": repo_url,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
    "use_fullscreen_button": True,
    "show_navbar_depth": 2,
    "navigation_with_keys": True,
    "toc_title": "On this page",
    # The lockup already carries the project name, so no text beside it.
    "logo": {
        "image_light": "_static/logo/dmt-logo.svg",
        "image_dark": "_static/logo/dmt-logo-dark.svg",
        "alt_text": "dmt - Dispersion Measure Transform",
    },
}
html_favicon = "_static/logo/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Autodoc / autosummary
# API pages use explicit autoclass/automodule directives, so do not ask
# autosummary to generate a file that is not in the tree.
autosummary_generate = False
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
suppress_warnings = ["autosummary.import_cycle"]


# -- Numpydoc
# Class members are rendered by autodoc. Leaving this on duplicates every
# method and emits "duplicate object description" warnings.
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "bool": "bool",
    "int": "int",
    "str": "str",
    "bytes": "bytes",
    "list": "list",
    "dict": "dict",
    "tuple": "tuple",
    "set": "set",
    "None": "None",
    "ndarray": "numpy.ndarray",
    "dtype": "numpy.dtype",
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
    "uint8": "numpy.uint8",
    "int32": "numpy.int32",
    "float32": "numpy.float32",
    "float64": "numpy.float64",
    "complex64": "numpy.complex64",
    "Path": "pathlib.Path",
    "Iterator": "collections.abc.Iterator",
    "Callable": "collections.abc.Callable",
    "Literal": "typing.Literal",
}
numpydoc_xref_ignore = {
    "of",
    "or",
    "shape",
    "type",
    "optional",
    "scalar",
    "default",
    "array_like",
    "array-like",
}

# MyST & MyST-NB configuration
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "substitution",
]
myst_heading_anchors = 3
myst_links_external_new_tab = True

# Notebooks in the repo already contain executed outputs (text and plots).
# MyST-NB renders those outputs when execution is off, which is what the
# HTML pages show. Set NB_EXECUTION_MODE=force to re-run them during a build.
nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "off")
nb_execution_timeout = 300
nb_execution_allow_errors = False
nb_execution_raise_on_error = True

# -- Sphinx copybutton
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_timeout = 10


def _pybind_overload_prose(lines: list[str]) -> list[str] | None:
    """Pull the hand-written section out of a pybind overload listing.

    Pybind concatenates every overload into one docstring, indents the
    prose, and separates overloads with a bare signature. That is not valid
    reStructuredText (``*`` starts emphasis, and the indented Parameters
    block unindents into the next overload). The prose is what we want
    numpydoc to format; autodoc already renders a signature.
    """
    if not any(line.strip() == "Overloaded function." for line in lines):
        return None
    prose: list[str] = []
    started = False
    for line in lines:
        if line.startswith((" ", "\t")) and line.strip():
            started = True
            prose.append(line)
            continue
        if started and not line.strip():
            prose.append("")
            continue
        if started:
            break
    if not any(line.strip() for line in prose):
        return None
    return textwrap.dedent("\n".join(prose)).splitlines()


def _escape_pybind_emphasis(
    _app: Any,
    _what: str,
    _name: str,
    _obj: Any,
    _options: Any,
    lines: list[str],
) -> None:
    """Make pybind docstrings safe for docutils before numpydoc sees them."""
    prose = _pybind_overload_prose(lines)
    if prose is not None:
        lines[:] = prose
        return
    markers = ("*args", "**kwargs", ", *,")
    for index, line in enumerate(lines):
        if not line.strip():
            lines[index] = ""
        elif any(marker in line for marker in markers):
            lines[index] = line.replace("*", r"\*")


def setup(app: Any) -> dict[str, bool]:
    # Priority below numpydoc (500) so this runs first. Ascending order.
    app.connect("autodoc-process-docstring", _escape_pybind_emphasis, priority=50)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Link Python objects to the matching lines on GitHub.

    C++ objects are covered by Breathe and have no Python source, so they
    resolve to ``None``. Pybind wrappers also have no inspectable source;
    those return ``None`` instead of aborting the build.
    """
    if domain != "py":
        return None
    module_name = info.get("module")
    fullname = info.get("fullname")
    if not module_name or not fullname:
        return None
    try:
        obj = importlib.import_module(module_name)
        for part in fullname.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        file_name = inspect.getsourcefile(obj)
        if not file_name:
            return None
        source, start = inspect.getsourcelines(obj)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None
    path = Path(file_name).resolve()
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    end = start + len(source) - 1
    return f"{repo_url}/blob/main/{rel.as_posix()}#L{start}-L{end}"


# -- Doxygen and Breathe -----------------------------------------------------


def _xml_has_compounds(xml_dir: Path) -> bool:
    index = xml_dir / "index.xml"
    if not index.is_file():
        return False
    # A placeholder index written when Doxygen is absent has no compounds.
    # Treating that as success skips regeneration and makes every
    # doxygenclass directive fail.
    head = index.read_text(encoding="utf-8", errors="ignore")[:8000]
    return "<compound " in head


def _candidate_xml_dirs() -> list[Path]:
    doxygen_out = os.environ.get("DOXYGEN_OUTPUT_DIR")
    candidates: list[Path] = []
    if doxygen_out:
        candidates.append(Path(doxygen_out) / "xml")
    candidates.extend(
        [
            DOCS_DIR / "_build" / "doxygen" / "xml",
            DOCS_DIR / "doxygen" / "xml",
            PROJECT_ROOT / "build" / "docs" / "doxygen" / "xml",
            PROJECT_ROOT / "_build" / "doxygen" / "xml",
        ]
    )
    return candidates


def _run_doxygen() -> None:
    doxyfile_path = DOCS_DIR / "Doxyfile"
    if not shutil.which("doxygen") or not doxyfile_path.exists():
        return
    try:
        print(f"Running doxygen from {DOCS_DIR}...")
        subprocess.run(
            ["doxygen", "Doxyfile"],
            cwd=str(DOCS_DIR),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Warning: Failed to execute doxygen automatically: {exc}")


def _doxygen_inputs_mtime() -> float:
    include = PROJECT_ROOT / "include"
    stamps = [(DOCS_DIR / "Doxyfile").stat().st_mtime]
    stamps.extend(path.stat().st_mtime for path in include.rglob("*") if path.is_file())
    return max(stamps)


def _doxygen_xml_is_current(xml_dir: Path) -> bool:
    index = xml_dir / "index.xml"
    if not _xml_has_compounds(xml_dir):
        return False
    return index.stat().st_mtime >= _doxygen_inputs_mtime()


def _prepare_breathe() -> Path | None:
    fresh = next(
        (path for path in _candidate_xml_dirs() if _doxygen_xml_is_current(path)),
        None,
    )
    if fresh is not None:
        return fresh
    _run_doxygen()
    produced = DOCS_DIR / "_build" / "doxygen" / "xml"
    if _xml_has_compounds(produced):
        return produced
    return next(
        (path for path in _candidate_xml_dirs() if _xml_has_compounds(path)), None
    )


def _doxygen_function(qualified_name: str, arguments: str = "") -> str:
    """One Breathe function directive.

    ``arguments`` is the Doxygen ``argsstring``, including parentheses.
    Overloaded names do not resolve without it.
    """
    return f".. doxygenfunction:: {qualified_name}{arguments}\n   :project: dmt\n"


def _convenience_functions_rst() -> str:
    linear = (
        "(std::span< const float > waterfall, float f_min, float f_max, "
        "SizeType nchans, SizeType nsamps, float tsamp, IndexType dt_max, "
        "IndexType dt_min=0, SizeType dt_step=1, bool use_box_smearing=true, "
        'std::string_view mode="valid", bool verbose=false, int nthreads=1, '
        "SizeType nbeams=1)"
    )
    dt_grid = (
        "(std::span< const float > waterfall, float f_min, float f_max, "
        "SizeType nchans, SizeType nsamps, float tsamp, "
        "const std::vector< IndexType > &dt_grid, "
        "bool use_box_smearing=true, "
        'std::string_view mode="valid", bool verbose=false, int nthreads=1, '
        "SizeType nbeams=1)"
    )
    dm_grid = (
        "(std::span< const float > waterfall, float f_min, float f_max, "
        "SizeType nchans, SizeType nsamps, float tsamp, "
        "const std::vector< float > &dm_grid, "
        "bool use_box_smearing=true, "
        'std::string_view mode="valid", bool verbose=false, int nthreads=1, '
        "SizeType nbeams=1)"
    )
    blocks = [
        _doxygen_function("dmt::algorithms::compute_fdmt", linear),
        _doxygen_function("dmt::algorithms::compute_fdmt", dt_grid),
        _doxygen_function("dmt::algorithms::compute_fdmt", dm_grid),
        _doxygen_function("dmt::algorithms::compute_fdmt_fft", linear),
        _doxygen_function("dmt::algorithms::compute_fdmt_fft", dt_grid),
        _doxygen_function("dmt::algorithms::compute_fdmt_fft", dm_grid),
        _doxygen_function("dmt::algorithms::add_frb_track"),
    ]
    return "\n".join(blocks)


def _write_cpp_api_body(*, has_xml: bool) -> None:
    """Write the Breathe fragment included by ``api/cpp_api.md``.

    The fragment is reStructuredText. When Doxygen XML is missing it contains
    only prose, so the build does not depend on the ``breathe`` extension.
    """
    if has_xml:
        body = """\
Compute Engines (``dmt::algorithms``)
--------------------------------------

.. doxygenclass:: dmt::algorithms::FDMTCPU
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::DDMTCPU
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::CohFDMTCPU
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::FDMTFFTCPU
   :project: dmt
   :members:

CUDA Engines
------------

.. doxygenclass:: dmt::algorithms::FDMTCUDA
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::DDMTCUDA
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::CohFDMTCUDA
   :project: dmt
   :members:

.. doxygenclass:: dmt::algorithms::FDMTFFTCUDA
   :project: dmt
   :members:

.. doxygenstruct:: dmt::algorithms::FDMTSubbandViewCUDA
   :project: dmt
   :members:

.. doxygenclass:: dmt::utils::DataUnpackerCUDA
   :project: dmt
   :members:

.. doxygenclass:: dmt::utils::FFTManagerCUDA
   :project: dmt
   :members:

Plans & Geometry (``dmt::plans``)
---------------------------------

.. doxygenclass:: dmt::plans::FDMTPlan
   :project: dmt
   :members:

.. doxygenclass:: dmt::plans::DDMTPlan
   :project: dmt
   :members:

.. doxygenclass:: dmt::plans::CohFDMTPlan
   :project: dmt
   :members:

.. doxygenstruct:: dmt::plans::FDMTComplexity
   :project: dmt
   :members:

.. doxygenstruct:: dmt::plans::FDMTShape
   :project: dmt
   :members:

.. doxygenstruct:: dmt::plans::FDMTCoord
   :project: dmt
   :members:

High-Level Convenience Functions
--------------------------------

@@FUNCTIONS@@

Types (``dmt/common/types.hpp``)
--------------------------------

.. doxygenfile:: types.hpp
   :project: dmt
"""
    else:
        body = """\
Doxygen XML was not available when this site was built, so the C++ symbols
are not expanded on this page.

Generate the XML, then rebuild the HTML docs:

.. code-block:: bash

   cd docs && doxygen Doxyfile
"""
    target = DOCS_DIR / "api" / "_breathe_body.rst"
    target.write_text(
        body.replace("@@FUNCTIONS@@", _convenience_functions_rst()),
        encoding="utf-8",
    )


breathe_xml_dir = _prepare_breathe()
_write_cpp_api_body(has_xml=breathe_xml_dir is not None)
if breathe_xml_dir is not None:
    extensions.append("breathe")
    breathe_projects = {"dmt": str(breathe_xml_dir)}
    breathe_default_project = "dmt"
    breathe_default_members = ("members", "undoc-members")
else:
    print(
        "Warning: Doxygen XML not found. The C++ API page will describe how "
        "to generate it instead of embedding Breathe directives."
    )
