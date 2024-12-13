from evaluation.code_execution import EvalResult, EvalTask, eval_results_to_markdown
from synthetic_data.utils import ensure_directory

GENERATED_SOLUTION = """
def filter_by_substring(strings: List[str], substring: str) -> List[str]:                                                                                                                         
    result = []                                                                                                                                                                                   
    for s in strings:                                                                                                                                                                             
        if substring in s:                                                                                                                                                                        
            result.append(s)                                                                                                                                                                      
    return result                                                                                                                                                                                 
"""
TEST_CODE = """

METADATA = {
    'author': 'jt',
    'dataset': 'test'
}

def check(candidate):
    assert candidate([], 'john') == []
    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxxAAA', 'xxx']
    assert candidate(['xxx', 'asd', 'aaaxxy', 'john doe', 'xxxAAA', 'xxx'], 'xx') == ['xxx', 'aaaxxy', 'xxxAAA', 'xxx']
    assert candidate(['grunt', 'trumpet', 'prune', 'gruesome'], 'run') == ['grunt', 'prune']
"""


def test_format_markdown():
    eval_result = EvalResult(
        prompt="Write a function to filter a list of strings by a given substring.",
        generated_code=GENERATED_SOLUTION,
        test=TEST_CODE,
        entry_point="filter_by_substring",
        err="",
        tests_pass=[True, True],
        task_id="function",
        task=EvalTask("fake", "fake", "humaneval", "ast"),
    )
    out = eval_results_to_markdown([eval_result, eval_result])
    ensure_directory("out")
    with open("out/test_format_markdown.md", "w") as f:
        f.write("\n".join(out))
