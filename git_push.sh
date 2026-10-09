#!/bin/bash
set -e

MESSAGE="${1:-Updated}"

git add .
git commit -m "$MESSAGE"
git push
