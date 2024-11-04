import sys

sys.path.append(".")

from synthetic_data.utils import extract_code_block

TEST_OUTPUT = """
Here's an example of structured data extraction from the given context in JSON form:

Query:
```json
{
  "entity": "Stanley J. Goldberg",
  "properties": [
    "birthYear",
    "birthPlace",
    "education",
    "careerMilestones"
  ]
}
```

Result:
```json
{
  "entity": "Stanley J. Goldberg",
  "data": {
    "birthYear": 1939,
    "birthPlace": "Maryland",
    "education": [
      {
        "degree": "B.S.",
        "institution": "University of Maryland, School of Business and Public Administration",
        "year": 1960
      },
      {
        "degree": "LL.B.",
        "institution": "University of Maryland School of Law",
        "year": 1964
      },
      {
        "type": "Graduate work",
        "subject": "Federal Income Taxation",
        "institution": "New York University"
      }
    ],
    "careerMilestones": [
      {
        "position": "Tax Attorney",
        "organization": "United States Department of Treasury, Office of Chief Counsel, Internal Revenue Service",
        "location": "New York City",
        "startDate": "January 1965"
      },
      {
        "position": "Special Trial Attorney",
        "year": 1976
      },
      {
        "position": "Assistant District Counsel",
        "year": 1984
      },
      {
        "position": "Special Trial Judge",
        "organization": "United States Tax Court",
        "appointmentDate": "August 4, 1985"
      }
    ]
  }
}
```

This query requests specific information about Stanley J. Goldberg, including his birth year, birth place, education, and career milestones. The resulting data provides factual information extracted from the context, organized according to the requested schema.
"""


def test_extract_json():
    json_schema = extract_code_block(TEST_OUTPUT)
    assert len(json_schema) == 2
