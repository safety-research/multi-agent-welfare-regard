#!/bin/sh
# Record exactly what is inside researcher-sandbox:v1 (image contents are part of the experimental condition).
{ echo "# researcher-sandbox:v1 contents ($(date -u +%F))"; echo; echo '## Python packages (pip freeze)'; echo '```';
  docker run --rm researcher-sandbox:v1 pip freeze; echo '```'; echo; echo '## System tools'; echo '```';
  docker run --rm researcher-sandbox:v1 sh -c 'python --version; git --version; rg --version | head -1; fd --version; jq --version; gcc --version | head -1';
  echo '```'; } > "$(dirname "$0")/CONTENTS.md"
