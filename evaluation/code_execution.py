import ast
import traceback
from typing import Any, Callable, List, Optional, Tuple
from rich.syntax import Syntax
from rich.console import Console
from typing import Optional, Callable
import ast
import contextlib
import io
import signal


from evaluation.python_interpereter import evaluate_python_code_ast, LIST_SAFE_MODULES

ALLOWED_FNS = {
    range,
    print,
    sum,
    enumerate,
    int,
    str,
    abs,
    zip,
    sorted,
    list,
    len,
    bin,
    isinstance,
    set,
    min,
    max,
    dict,
    filter,
    reversed,
    chr,
    float,
    ord,
    tuple,
    bool,
    map,
    round,
}
ALLOWED_FN_DICT = {fn.__name__: fn for fn in ALLOWED_FNS}


class AssertToBoolTransformer(ast.NodeTransformer):
    def __init__(self, result_list_name="results"):
        self.result_list_name = result_list_name
        self.n_asserts = 0

    def visit_FunctionDef(self, node):
        list_init = ast.Assign(
            targets=[ast.Name(id=self.result_list_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )

        ast.copy_location(list_init, node.body[0])

        node.body.insert(0, list_init)
        self.generic_visit(node)

        return_stmt = ast.Return(
            value=ast.Name(id=self.result_list_name, ctx=ast.Load())
        )
        ast.copy_location(return_stmt, node.body[-1])
        node.body.append(return_stmt)
        return node

    def visit_Assert(self, node):
        append_expr = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.result_list_name, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Call(
                        func=ast.Name(id="bool", ctx=ast.Load()),
                        args=[node.test],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )
        self.n_asserts += 1

        return ast.copy_location(append_expr, node)


def assertions_to_tests(test_code: str, entrypoint: str):
    test_code = test_code.replace("candidate(", entrypoint + "(")

    tree = ast.parse(test_code)

    transformer = AssertToBoolTransformer("results")
    transformed_tree = transformer.visit(tree)

    transformed_code = ast.unparse(transformed_tree)
    return transformed_code, transformer.n_asserts


def print_code_snippet(snippet: str, console: Console):
    formatted_snippet = Syntax(
        snippet,
        "python",
        theme="monokai",
        line_numbers=True,
    )
    console.print(formatted_snippet)


ALLOWED_IMPORTS = LIST_SAFE_MODULES + [
    "typing",
    "copy",
    "hashlib",
    "string",
    "collections",
]


def evaluate_sample_humaneval(
    sample: str, solution: str, tests: str, entrypoint: str
) -> Tuple[Optional[str], List]:
    """
    Evaluate a code snippet against a set of tests.
    Returns an error message and a list of test results.
    """
    tests, n_asserts = assertions_to_tests(tests, entrypoint)
    full_code = sample + solution + tests + "\ncheck()"
    try:
        fn_out = evaluate_python_code_ast(
            full_code,
            ALLOWED_FN_DICT,
            authorized_imports=ALLOWED_IMPORTS,
        )
        return None, fn_out  # type: ignore
    except Exception as e:
        traceback.print_exc()
        return str(e), [False] * n_asserts


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def evaluate_python_code_exec(
    code_to_run: str, test_inputs: str, timeout: float = 1000
) -> Tuple[Optional[str], Any]:

    result = []
    inputs_idx = 0
    input_values = []

    test_inputs_list = test_inputs.strip().split("\n")
    code_to_run = code_to_run + f"\nresult = solution({test_inputs_list})"

    def _retrieve_input(value=None):
        nonlocal inputs_idx
        out = input_values[inputs_idx]
        inputs_idx += 1
        return out

    def _exit(value=None):
        return

    local_vars = {}

    try:
        exec_globals = {
            "input": _retrieve_input,
            "exit": _exit,
        }
        with swallow_io():
            with time_limit(timeout):
                exec(code_to_run, exec_globals, local_vars)
        output = local_vars.get("result")
        return None, output
    except TimeoutException:
        return "timed out", None
    except Exception as e:
        traceback.print_exc()
        return str(e), None


def evaluate_sample_codecontests(
    code_to_run: str, input_fn: Callable
) -> Tuple[Optional[str], List]:
    try:
        fn_out = evaluate_python_code_ast(
            code_to_run,
            ALLOWED_FN_DICT,
            authorized_imports=ALLOWED_IMPORTS,
            custom_tools={"input": input_fn},
        )
        return None, fn_out  # type: ignore
    except Exception as e:
        return str(e), []
