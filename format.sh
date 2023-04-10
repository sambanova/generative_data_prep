#!/bin/bash
set -eo pipefail

format() {
    # TODO: Automatically add license to the top of each file.
    PYTHON_FILES=()
    while IFS=$'\n' read -r line; do PYTHON_FILES+=("$line"); done \
    < <($1 -- '*.py')
    for i in "${!PYTHON_FILES[@]}"; do
        # If file does not exist move on (e.g. 'git mv')
        if [ ! -f "${PYTHON_FILES[i]}" ]; then
            continue
        fi

        echo "Formatting...${PYTHON_FILES[i]}"

        # NOTE: yapf->autoflake->isort order needs to be consistent with check_format.sh
        yapf -i "${PYTHON_FILES[i]}" && autoflake --in-place --remove-all-unused-imports --remove-unused-variables ${PYTHON_FILES[i]} --exclude=__init__.py
        ret=$?
        if [[ $ret != 0 ]]; then
            echo "ERROR formatting ${PYTHON_FILES[i]}"
            exit 1
        fi

        isort "${PYTHON_FILES[i]}"
        ret=$?
        if [[ $ret != 0 ]]; then
            echo "ERROR sorting the import order ${PYTHON_FILES[i]}"
            exit 1
        fi
    done
}

# Ensure the changed files are properly formatted.
echo "---YAPF FORMATTING CHANGED PYTHON FILES---"
format "git diff --name-only origin/main..."
format "git diff --name-only"
