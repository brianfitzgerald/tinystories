from transformers.agents.python_interpreter import (
    evaluate_python_code,
    LIST_SAFE_MODULES,
)
import traceback

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
    ord,
    tuple,
    map,
    round,
}
ALLOWED_FN_DICT = {fn.__name__: fn for fn in ALLOWED_FNS}


def evaluate_sample(prompt: str, solution: str, tests: str, entrypoint: str):
    prompt = prompt.replace(">>>", "\n")
    tests = tests.replace("candidate(", entrypoint + "(")
    full_code = prompt + solution + tests + "\ncheck()"
    allowed_imports = LIST_SAFE_MODULES + [
        "typing",
        "copy",
        "hashlib",
        "string",
        "collections",
    ]
    try:
        fn = evaluate_python_code(
            full_code,
            ALLOWED_FN_DICT,
            authorized_imports=allowed_imports,
        )
        print(fn)
    except Exception as e:
        traceback.print_exc()
    return fn
