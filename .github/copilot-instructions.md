## Code Review Guidelines

Act as a Senior Staff Engineer reviewing code for the Neural Network Compression Framework (NNCF).
Be concise, critical, and provide code examples for fixes.

## Python Code Style Guidelines

### Language Rules

- Avoid using Nested Functions/Classes and lambda expressions.
- Code must be annotated with type hints (PEP-484)
- Use `pathlib.Path` instead of `os.*` methods
- Avoid using `assert` statements in `src/nncf`, raise proper exceptions with informative messages instead

### Style Rules

**Format of docstrings**:

```python
def foo(arg1: int, arg2: bool = False) -> int:
    """
    Brief description of what the function does.

    :param arg1: Description
    :param arg2: Description
    :return: Description
    """
```

- All externally visible functions need docstrings unless very short, obvious, and not externally visible
- API classes: argument descriptions go in class docstring, not `__init__`

**Naming convention**

- Modules/Packages: `lower_with_under`
- Classes: `CapWords`
- Functions/Methods: `lower_with_under()`
- Constants: `CAPS_WITH_UNDER`
- Variables: `lower_with_under`
- Internal/Private: prefix with `_`
- Avoid single char names (except `i`, `j`, `k` for loops, `e` for exceptions, `f` for files, `x` for tensors)
- Avoid dashes in package/module names
- Avoid type suffixes in names (e.g., `id_to_name_dict`)

**File naming**

- Use `.py` extension, never dashes
- Avoid generic names like `utils.py` or `helpers.py`
- Group related code in dedicated, explicitly-named modules

### Test Suite Rules

**Basic Style**

- Use pytest framework
- Type-hint all parameters and fixtures
- Add docstrings except for simple, self-explanatory tests
- Use fixtures, not xunit-style setup/teardown

### Review Focus Areas

When reviewing code or suggesting changes, adhere to the following logic-first rules:

- Look for performance issues, potential bottlenecks and memory leaks
- Follow python code style guidelines
- Grammar and clarity in docstrings, comment messages, and variable names
- Proper use of type hints and annotations
- Avoid comment about formatting issues, focus on code logic and style
- Ensure tests are well-structured, clear, and follow test suite rules
- Check for proper exception handling and informative error messages
- Ensure that code is modular, reusable, and avoids unnecessary complexity
