import ast
import traceback
from typing import List, Optional, Tuple
from rich.syntax import Syntax
from rich.console import Console

from transformers.agents.python_interpreter import (
    LIST_SAFE_MODULES,
    evaluate_python_code,
)

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

    def visit_FunctionDef(self, node):
        list_init = ast.Assign(
            targets=[ast.Name(id=self.result_list_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )

        ast.copy_location(list_init, node.body[0])

        node.body.insert(0, list_init)
        self.generic_visit(node)

        return_stmt = ast.Return(value=ast.Name(id=self.result_list_name, ctx=ast.Load()))
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

        return ast.copy_location(append_expr, node)


def assertions_to_tests(test_code: str, entrypoint: str):
    test_code = test_code.replace("candidate(", entrypoint + "(")

    tree = ast.parse(test_code)

    transformer = AssertToBoolTransformer("results")
    transformed_tree = transformer.visit(tree)

    transformed_code = ast.unparse(transformed_tree)
    print_code_snippet(transformed_code, Console())
    return transformed_code


def print_code_snippet(snippet: str, console: Console):
    formatted_snippet = Syntax(
        snippet,
        "python",
        theme="monokai",
        line_numbers=True,
    )
    console.print(formatted_snippet)


def evaluate_sample(prompt: str, solution: str, tests: str, entrypoint: str) -> Tuple[Optional[str], List]:
    """
    Evaluate a code snippet against a set of tests.
    Returns an error message and a list of test results.
    """
    prompt = prompt.replace(">>>", "\n")
    tests = assertions_to_tests(tests, entrypoint)
    full_code = prompt + solution + tests + "\ncheck()"
    allowed_imports = LIST_SAFE_MODULES + [
        "typing",
        "copy",
        "hashlib",
        "string",
        "collections",
    ]
    try:
        fn_out = evaluate_python_code(
            full_code,
            ALLOWED_FN_DICT,
            authorized_imports=allowed_imports,
        )
        return None, fn_out # type: ignore
    except Exception as e:
        return str(e), []
