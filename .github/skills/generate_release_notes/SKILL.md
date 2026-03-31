---
name: generate-release-notes
description: Generate release notes for the new NNCF release.
---

# Generate release notes for NNCF

## Description

Automatically generates release notes for the new NNCF release based on the commits since the last release. The generated release notes should be formatted in a clear and concise manner, excluding commits from dependabot and other automated tools.
Release branch detected by the format `release_v*` will be used to determine the commits for the release notes.
## Context

**File Paths:**

- `ReleaseNotes.md`
- `src/nncf/version.py`

## Instructions

### Constraints

- To collect diff between the last two release branches, use only provided code snippet in Step 1. Do not use any other git commands to collect the commits.

### Step 1. Run follow code to collect commits since the last release in the `commits.txt` file

```bash
# 1. Get the two most recent release branches from origin
git fetch origin
BRANCHES=$(git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/**/release_v* | head -n 2)

# 2. Assign them to variables (Current is the 1st, Previous is the 2nd)
LAST=$(echo "$BRANCHES" | sed -n '1p')
PREV=$(echo "$BRANCHES" | sed -n '2p')

echo "Comparing $PREV to $LAST..."

# 3. Generate the log and filter out dependabot in one pipeline
git log "$PREV..$LAST" --pretty=format:"%an;%s" | grep -v "dependabot" > commits.txt
```

### Step 2. Read current NNCF version

Checkout to the `$LAST` release branch and read the current version of NNCF from `src/nncf/version.py`. This will be used to determine the new version number for the release notes. The version is typically defined in the format `__version__ = "X.Y.Z"`.

### Step 3. Change target branch of the pull request

- If the pull request for the release notes was already created, change the target branch to the `$LAST` release branch from previous step.
For example, if the release branch is `release_v230`, change the target branch of the pull request to `release_v230`.

### 4. Generate release notes

- Based on the collected commits in `commits.txt`, generate release notes in the `ReleaseNotes.md` file.
- The release notes should be formatted in a clear and concise manner, using formats from the previous releases as a reference.
- Add a link to the pull request for each change in the end of line in the format `(#PR_NUMBER)`. For example, if the commit message is "Fix bug in NNCF (#123)", the release notes should include "Fix bug in NNCF (#123)".
- Ensure that the release notes are comprehensive and accurately reflect the changes made since the last release, while maintaining readability and clarity for users.
