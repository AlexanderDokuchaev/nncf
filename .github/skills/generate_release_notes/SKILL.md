---
name: generate-release-notes
description: Generate release notes for the new NNCF release.
---

# Generate release notes for NNCF

## Context

**File Paths:**

- `ReleaseNotes.md`
- `src/nncf/version.py`
- `.github/skills/generate_release_notes/parse_commits.sh`

## Instructions

## Execution Logic
1.  **Execute**: Run the local file `.github/skills/generate_release_notes/parse_commits.sh`.
2. **Read**: Open the generated `commits.txt`.
3. **Read**: Read the current version of NNCF from `src/nncf/version.py` to determine the new version number for the release notes.
4. **Updater**: Update `ReleaseNotes.md` with the new release notes based on the collected commits in `commits.txt`. The release notes should be formatted in a clear and concise manner, using formats from the previous releases as a reference. Add a link to the pull request for each change in the end of line in the format `(#PR_NUMBER)`. For example, if the commit message is "Fix bug in NNCF (#123)", the release notes should include "Fix bug in NNCF (#123)".


