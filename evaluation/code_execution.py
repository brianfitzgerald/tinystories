import ast
import traceback
from typing import Any, Callable, List, Optional, Tuple
from rich.syntax import Syntax
from rich.console import Console
from typing import Optional, Callable, Literal, Dict
import ast
import contextlib
import io
from enum import Enum
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from wrapt_timeout_decorator import timeout
from dataclasses import dataclass
from abc import ABC
from loguru import logger
import re
import signal

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
    "functools",
]


def evaluate_sample_ast(
    full_code: str, n_asserts: int
) -> Tuple[Optional[str], List]:
    """
    Evaluate a code snippet via the AST interpreter.
    Returns an error message and a list of test results.
    """

    try:
        allowed_fns = {**ALLOWED_FN_DICT}
        fn_out = evaluate_python_code_ast(
            full_code,
            allowed_fns,
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
def evaluate_sample_exec(
    code_to_run: str, inputs_list: List[str]
) -> Tuple[Optional[str], Optional[List]]:
    """
    Evaluate generated code with a list of inputs and outputs.
    The code is run with multiple inputs, each being used when the code calls `input()`.
    """

    inputs_idx = 0

    def _retrieve_input(value=None):
        nonlocal inputs_idx
        out = inputs_list[inputs_idx]
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
                with time_limit(2):
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
CodeExecutionMode = Literal["ast", "exec"]
# Format to use for prompting and dataset style
CodeTaskFormat = Literal["humaneval", "mbpp"]


@dataclass
class EvalTask(ABC):
    name: str
    dataset_uri: str
    code_task_format: Optional[CodeTaskFormat]
    code_execution_mode: Optional[CodeExecutionMode]
    eval_split: str = "test"


@dataclass
class EvalResult:
    prompt: str
    generated_code: str
    test: str
    entry_point: str
    err: Optional[str]
    tests_pass: List[bool]
    task_id: str
    task: EvalTask


def evaluate_codecontests(
    console: Console, generation_results: list[tuple[str, dict]], eval_task: EvalTask
) -> List[EvalResult]:
    results_batch: List[EvalResult] = []
    for result, sample_dict in generation_results:
        for generated in result:
            full_code = generated
            code_snippets = extract_code_block(generated, "python")
            if len(code_snippets) == 0:
                logger.warning("No code snippets found in generated code")
                continue
            generated_code = code_snippets[0]
            if len(code_snippets) > 1:
                logger.warning("Multiple code snippets found in generated code")
            if eval_task.code_task_format == "mbpp":
                mbpp_problem = MBPPProblem(**sample_dict)
                sample = _convert_mbpp_to_humaneval(mbpp_problem)
                tests, n_asserts = assertions_to_tests(sample.test, sample.entry_point)
                n_asserts = len(mbpp_problem.test_list)
            elif eval_task.code_task_format == "humaneval":
                sample = HumanEvalProblem(**sample_dict)
                tests, n_asserts = assertions_to_tests(sample.test, sample.entry_point)
            full_code = generated_code + "\n" + tests + "\ncheck()"
            console.print(f"Evaluating sample: {sample.task_id}")
            console.print(f"Canonical solution:")
            print_code_snippet(sample.canonical_solution, console)
            prompt = "" if sample.prompt == "text" else sample.prompt
            exec_err, evaluation_results = evaluate_sample_ast(
                full_code, n_asserts
            )
            console.print(f"Generated solution:")
            print_code_snippet(generated_code, console)
            console.print(f"Test code:")
            print_code_snippet(sample.test, console)
            _print_test_results(exec_err, evaluation_results, console)
            console.print("=" * console.size.width)
            results_batch.append(
                EvalResult(
                    prompt,
                    generated_code,
                    sample.test,
                    sample.entry_point,
                    exec_err,
                    evaluation_results,
                    sample.task_id,
                    eval_task,
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
        pass_checks = ["✓" if passed else "✗" for passed in er.tests_pass]
        md_lines.extend(" ".join(pass_checks))
        md_lines.extend(["", "---", ""])

    return md_lines


@dataclass
class CodeContestsProblem:
    source: int
    difficulty: int
    name: str
    description: str
    public_tests: Dict
    private_tests: Dict
    cf_rating: int
    cf_points: float
    solution: Optional[str] = None


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@dataclass
class MBPPProblem:
    task_id: str
    text: str
    code: str
    test_list: List[str]
    test_setup_code: str
    challenge_test_list: List[str]


def get_fn_name_from_assert(code: str):
    match = re.search(r"\bassert\s+([a-zA-Z_]\w*)\s*\(", code)
    return match.group(1) if match else None


def _convert_mbpp_to_humaneval(sample: MBPPProblem) -> HumanEvalProblem:
    test_code = "\n\t".join(sample.test_list)
    test_code = f"def check():\n\t{test_code}"
    return HumanEvalProblem(
        task_id=sample.task_id,
        prompt=sample.text,
        canonical_solution=sample.code,
        test=test_code,
        entry_point=sample.code,
    )

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@timeout(5)
def evaluate_sample_against_codecontests_tests(
    completion: str,
    test_inputs: List[str],
    test_outputs: List[str],
    execution_mode: CodeExecutionMode,
) -> Tuple[List[List[str]], List[bool]]:
    """
    Evaluate a code snippet against a list of test inputs and outputs.
    Used for synthetic evaluation.
    """
    test_results_for_completion = []
    test_case_has_errors = []
    for test_input, expected_output_str in zip(test_inputs, test_outputs):
        expected_output = expected_output_str.strip().split("\n")
        inputs_list = test_input.strip().split("\n")
        if execution_mode == "exec":
            err, execution_output = evaluate_sample_exec(completion, inputs_list)
        elif execution_mode == "ast":
            full_code = completion + "\n" + test_input
            err, execution_output = evaluate_sample_ast(
                full_code, len(expected_output)
            )
        logger.info(
            f"Test output for completion: {execution_output}, expected: {expected_output}"
        )
        if err is not None:
            logger.info(
                f"Error in test case execution - error: {err}, results: {execution_output}"
            )
            test_results_for_completion.append([False] * len(expected_output))
            test_case_has_errors.append(True)
            continue
        if not isinstance(execution_output, list):
            logger.info(f"Expected list of outputs, got: {type(execution_output)}")
            test_results_for_completion.append([False] * len(expected_output))
            continue
        elif len(execution_output) != len(expected_output):
            logger.info(
                f"Length of execution outputs does not match no. of test cases: expected {len(expected_output)}, actual: {len(execution_output)}"
            )
            test_results_for_completion.append([False] * len(expected_output))
            continue
        else:
            test_case_results = []
            for expected, actual in zip(expected_output, execution_output):
                test_case_results.append(str(expected) == str(actual))
            test_results_for_completion.append(test_case_results)
        test_case_has_errors.append(False)
    return test_results_for_completion, test_case_has_errors
