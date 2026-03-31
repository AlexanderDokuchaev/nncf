#!/bin/bash
set -euxo pipefail

# 1. Get the two most recent release branches from origin
git fetch origin 'refs/heads/release_v*:refs/remotes/origin/release_v*'
git for-each-ref --sort=-committerdate --format="%(refname:short)" refs/**/release_v* | head -n 2 > branches_tmp.txt

# 2. Assign them to variables (Current is the 1st, Previous is the 2nd)
LAST=$(sed -n '1p' branches_tmp.txt)
PREV=$(sed -n '2p' branches_tmp.txt)

echo "Comparing $PREV to $LAST..."

# 3. Generate the log and filter out dependabot in one pipeline
git log "$PREV..$LAST" --pretty=format:"%an;%h;%s" | grep -v "dependabot" > tmp_release_commits.txt

rm branches_tmp.txt

cat tmp_release_commits.txt
