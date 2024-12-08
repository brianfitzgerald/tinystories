import ast
import traceback
from typing import Any, Callable, List, Optional, Tuple
from rich.syntax import Syntax
from rich.console import Console
from typing import Optional, Callable, Literal
import ast
import contextlib
import io
import signal
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from wrapt_timeout_decorator import timeout
from dataclasses import dataclass
from abc import ABC

from evaluation.python_interpereter import evaluate_python_code_ast, LIST_SAFE_MODULES
from synthetic_data.utils import extract_code_block

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

# CodeContests definitions


class Language(Enum):
    UNKNOWN = 0
    PYTHON = 1
    CPP = 2
    PYTHON3 = 3
    JAVA = 4


class ProblemSource(Enum):
    UNKNOWN = 0
    CODECHEF = 1
    CODEFORCES = 2
    HACKEREARTH = 3
    CODEJAM = 4
    ATCODER = 5
    AIZU = 6


class Difficulty(Enum):
    UNKNOWN_DIFFICULTY = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    HARDER = 4
    HARDEST = 5
    EXTERNAL = 6
    A = 7
    B = 8
    C = 9
    D = 10
    E = 11
    F = 12
    G = 13
    H = 14
    I = 15
    J = 16
    K = 17
    L = 18
    M = 19
    N = 20
    O = 21
    P = 22
    Q = 23
    R = 24
    S = 25
    T = 26
    U = 27
    V = 28


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


class RemoveMetadataTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        # Check if the target of the assignment is METADATA
        if any(
            isinstance(target, ast.Name) and target.id == "METADATA"
            for target in node.targets
        ):
            return None  # Remove this node
        return node  # Keep other nodes


def assertions_to_tests(test_code: str, entrypoint: str):
    test_code = test_code.replace("candidate(", entrypoint + "(")

    tree = ast.parse(test_code)

    transformer = AssertToBoolTransformer("results")
    transformed_tree = transformer.visit(tree)
    transformed_tree = RemoveMetadataTransformer().visit(transformed_tree)

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
    full_code = sample + solution + "\n" + tests + "\ncheck()"
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
        print(args, kwargs)

    def readline(self, *args, **kwargs):
        print(args, kwargs)

    def readlines(self, *args, **kwargs):
        print(args, kwargs)

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


@contextmanager
def swallow_io():
    output = io.StringIO()
    with redirect_stdout(output):
        yield output.getvalue()
    output.close()


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@timeout(2)
def evaluate_python_code_exec(
    code_to_run: str, test_inputs: str, timeout_sec: float = 10
) -> Tuple[Optional[str], Any]:

    inputs_idx = 0
    input_values = []

    test_inputs_list = test_inputs.strip().split("\n")
    input_values = test_inputs_list

    def _retrieve_input(value=None):
        nonlocal inputs_idx
        out = input_values[inputs_idx]
        inputs_idx += 1
        return out

    def _exit(value=None):
        return

    try:
        exec_globals = {
            "input": _retrieve_input,
            "exit": _exit,
        }
        full_globals = globals().copy()
        full_globals.update(exec_globals)

        out_io = io.StringIO()
        with redirect_stdout(out_io):
            with redirect_stderr(out_io):
                exec(code_to_run, full_globals)
        output_values = out_io.getvalue().strip().split("\n")
        try:
            output_values = [ast.literal_eval(val) for val in output_values]
        except Exception:
            pass
        return None, output_values
    except TimeoutError:
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


def _print_test_results(err: Optional[str], results: List[bool], console: Console):
    result_str = "Results: "
    if err:
        result_str += f"[red]Execution error: {err}[/red]"
    else:
        passed = sum(results)
        total = len(results)
        result_str += f"{passed}/{total} tests passed"
    console.print(result_str)


# Whether to evaluate via AST interpereter or exec() function
CodeEvalType = Literal["ast", "exec"]
# Format to use for prompting and dataset style
CodeTaskFormat = Literal["humaneval", "mbpp"]


@dataclass
class EvalTask(ABC):
    name: str
    dataset_uri: str
    code_task_format: Optional[CodeTaskFormat]
    code_eval_type: Optional[CodeEvalType]
    eval_split: str = "test"


@dataclass
class EvalResult:
    prompt: str
    generated_code: str
    test: str
    entry_point: str
    err: Optional[str]
    tests_pass: List[bool]


def evaluate_codecontests(
    console: Console, results: list[tuple[str, dict]], eval_task: EvalTask
) -> List[EvalResult]:
    results_batch: List[EvalResult] = []
    for result, sample in results:
        for generated in result:
            console.print(f"Function name: {sample['entry_point']}")
            console.print(f"Canonical solution:")
            print_code_snippet(sample["canonical_solution"], console)
            generated_code = extract_code_block(generated, "python")[0]
            exec_err, evaluation_results = evaluate_sample_humaneval(
                sample["prompt"],
                generated_code,
                sample["test"],
                sample["entry_point"],
            )
            console.print(f"Generated solution:")
            print_code_snippet(generated_code, console)
            console.print(f"Test code:")
            print_code_snippet(sample["test"], console)
            _print_test_results(exec_err, evaluation_results, console)
            console.print("=" * console.size.width)
            results_batch.append(
                EvalResult(
                    sample["prompt"],
                    generated_code,
                    sample["test"],
                    sample["entry_point"],
                    exec_err,
                    evaluation_results,
                )
            )
    return results_batch


def eval_results_to_markdown(evalresults: List[EvalResult]) -> List[str]:
    """
    Convert a list of EvalResult objects to a file string.
    Returns the individual lines of the file.
    """
    md_lines = []
    for i, er in enumerate(evalresults, start=1):
        md_lines.extend(
            [
                f"## Result {i}: {er.entry_point}",
                "",
                f"**Prompt:** {er.prompt}",
                "",
                "**Generated Code:**",
                "```python",
                er.generated_code.strip(),
                "```",
                "",
                "**Test Code:**",
                "```python",
                er.test.strip(),
                "```",
                "",
                "**Code Snippet:**",
                "```python",
                er.generated_code.strip(),
                "```",
                "**Error:**" if er.err else "**Error:** No error",
            ]
        )
        if er.err:
            md_lines.extend(["```", er.err.strip(), "```"])
        md_lines.extend(["", "**Tests Passed:**"])
        md_lines.extend(
            [
                f"- Test {idx}: {'✓' if passed else '✗'}"
                for idx, passed in enumerate(er.tests_pass, start=1)
            ]
        )
        md_lines.extend(["", "---", ""])

    return md_lines
