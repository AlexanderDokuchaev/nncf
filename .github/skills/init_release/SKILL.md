---
name: init-release
description: |
    Initialize a new release cycle, create all necessary branches and pull requests.
    Usages:
        gh copilot --allow-all-tools -p "Run init-release"
---

General instructions for the init-release skill:

- Stop on any error and print all executed commands to the console for better debugging.

### Step 1. Parse current version of NNCF and init env variables

```bash
#!/bin/bash
set -euxo pipefail

RELEASE_VERSION=$(grep '__version__' src/nncf/version.py | sed 's/.*"\([^"]*\)".*/\1/')
VERSION_NO_DOT=$(echo ${RELEASE_VERSION} | tr -d '.')
RELEASE_BRANCH="release_v${VERSION_NO_DOT}"

echo "Current version: ${RELEASE_VERSION}"
echo "Release branch: ${RELEASE_BRANCH}"
```

### Step 2. Create release branch from develop and push to origin

```bash
#!/bin/bash
set -euxo pipefail

git checkout develop
git pull origin develop
git checkout -b ${RELEASE_BRANCH}
git push origin ${RELEASE_BRANCH}
```

### Step 3. Create pull request to increase MINOR version of package

```bash
#!/bin/bash
set -euxo pipefail

git checkout develop
git pull origin develop

git checkout -b bump_version_v${VERSION_NO_DOT}

# Bump minor version
IFS='.' read -r -a VERSION_PARTS <<< "${RELEASE_VERSION}"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
NEW_MINOR=$((MINOR + 1))
NEW_VERSION="${MAJOR}.${NEW_MINOR}.0"

sed -i "s/__version__ = \"${RELEASE_VERSION}\"/__version__ = \"${NEW_VERSION}\"/" src/nncf/version.py
git commit -am "Bump NNCF version to ${NEW_VERSION}"
git push origin bump_version_v${VERSION_NO_DOT}

# Create pull request (using GitHub CLI)
gh pr create \
--title "Bump NNCF version to ${NEW_VERSION}" \
--body "This PR bumps the NNCF version to ${NEW_VERSION} on the develop branch." \
--base develop
```

### Step 4. Collect commits since the last released tag

```bash
#!/bin/bash
set -euxo pipefail
git checkout ${RELEASE_BRANCH}
git pull origin ${RELEASE_BRANCH}

PREV_RELEASE_BRANCH=$(git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/remotes/origin/release_v* | sed -n '2p')
git fetch origin ${PREV_RELEASE_BRANCH}:${PREV_RELEASE_BRANCH}

git log ${PREV_RELEASE_BRANCH_HASH}..${RELEASE_BRANCH} --pretty=format:"%an;%s" > commits.txt
grep -v "dependabot" commits.txt > filtered_commits.txt
```

### Step 5. Generate release notes

- Based on the collected commits in `filtered_commits.txt`, generate release notes in the `RELEASE_NOTES.md`:
- use formats from the previous releases as a reference.
- add link to the pull request for each change in the end of line in the format `(#PR_NUMBER)`. For example, if the commit message is "Fix bug in NNCF (#123)", the release notes should include "Fix bug in NNCF (#123)".

### Step 6. Create pull request with release notes

```bash
#!/bin/bash
set -euxo pipefail

RELEASE_NOTES_BRANCH="r${VERSION_NO_DOT}/release_notes"

git checkout ${RELEASE_BRANCH}
git checkout -b ${RELEASE_NOTES_BRANCH}
git add RELEASE_NOTES.md
git commit -m "Add release notes for NNCF version ${RELEASE_VERSION}"
git push origin ${RELEASE_NOTES_BRANCH}

gh pr create \
--title "Release notes for NNCF version ${RELEASE_VERSION}" \
--body "This PR adds release notes for NNCF version ${RELEASE_VERSION}." \
--base ${RELEASE_BRANCH} \
--head ${RELEASE_NOTES_BRANCH}

BODY_TEXT="Release notes based on commits since last release:\n

\`\`\`
$(cat filtered_commits.txt | sort)
\`\`\`"

gh pr comment --body "$BODY_TEXT"
```
