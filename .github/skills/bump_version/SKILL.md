---
name: bump-version
description: Increment the version of NNCF in src/nncf/version.py.
---

# Bump version of NNCF

## Description

Automatically updates the version string in `src/nncf/version.py`. This skill ensures that version bumps follow a consistent semantic logic, defaulting to a Minor version increment.


# Skill: Auto-Increment NNCF Version

## Context

- **File Path:** `src/nncf/version.py`
- **Variable Name:** `__version__`
- **Format:** Standard Semantic Versioning (`"Major.Minor.Patch"`)

## Instructions

### 1. Version Detection

Locate the line defining the version in `src/nncf/version.py` by searching for the pattern `__version__ = "X.Y.Z"`, where `X`, `Y`, and `Z` are integers representing the Major, Minor, and Patch versions respectively.

### 2. Increment Logic

Update version based on the user's intent. The user may specify which component to increment (Major, Minor, or Patch). If the intent is ambiguous, default to incrementing **the Minor version**.

| Component | Action                                      | Example (from 1.2.3) |
| :---      | :------------------------------------------ | :---                 |
| **Major** | Increment Major, reset Minor and Patch to 0 | `2.0.0`              |
| **Minor** | **Increment Minor, reset Patch to 0**       | **`1.3.0`**          |
| **Patch** | Increment Patch only                        | `1.2.4`              |


### 3. Execution Constraints

- Do not modify any other variables or imports within `src/nncf/version.py`.
- Ensure the version string remains wrapped in double quotes.

## Examples

**User:** "Bump version of NNCF"
**Copilot:**
- Reads `src/nncf/version.py` -> `__version__ = "2.4.1"`
- Updates to -> `__version__ = "2.5.0"`
