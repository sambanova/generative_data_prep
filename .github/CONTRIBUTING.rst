##################
Contributing Guide
##################

#. Open a Pull Request (PR) to submit any code changes
#. Please make sure your local environment is set-up properly to run preliminary checks

Local Environment Setup
***********************

#. Create Python virtual environment ``venv`` in the root of local software-repository

    .. code-block::

        python -m venv venv
        source venv/bin/activate

#. Install and set-up Pre-commit

    .. code-block::

        pip install pre-commit
        pre-commit install


Naming Conventions
******************

#. git branch naming convention

   - ``<username>/<feature/bugfix/hotfix>/<a-short-and-clear-description>``

   - e.g. ``john/feature/json-tests-should-support-iommu``


Pull Request (PR) Process
*************************

#. Ensure ``pre-commit`` is running with the repository configuration before opening a PR
#. A PR should only contain one unit of work; please open multiple PR's as necessary
#. Do your best to make sure all PR checkboxes could be ticked off
#. The PR should pass all the automated checks before it could be merged

Pull Request (PR) Review
************************

#. If you are assigned to review a PR, respond as soon as possible
   - If you are not the right person to be reviewing the PR, please find another relevant person from your team and assign it to them
#. Provide actionable explicit comments with code-examples if possible
#. For soft suggestions use prefix ``nit:`` in your comments
#. Use ``Start Review`` feature to submit multiple comments at once.
#. Use ``Request Changes`` to block the PR explicitly until the questions/concerns are resolved.

Code of Conduct
***************

#. When reviewing PR, imagine yourself as a PR submitter
#. When responding to PR feedback, imagine yourself as a PR reviewer
#. Be honest, direct, and respectful in your communication; embrace difference of opinions
#. For any comments that is going through many back and forths; hop on a quick-call to understand the other persons viewpoint
