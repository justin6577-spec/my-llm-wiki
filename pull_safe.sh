#!/bin/bash
echo "Pushing server changes first..."
git add .
git diff --cached --quiet || git commit -m "Auto-save before pull sync"
git push origin main
echo "Done. Now run git pull on your local PC."
