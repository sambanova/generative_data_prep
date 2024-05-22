##################
Contributing Guide
##################

#. Open a Pull Request (PR) to submit any code changes
#. Please make sure your local environment is set-up properly to run preliminary checks

Local Environment Setup
***********************

**NOTE**: Please ensure that your Python & Docker version matches the version used in CI flow in `VERSIONS <../.circleci/VERSIONS>`_ file.

#. Running in ``Docker`` container

   For repeatability, it's better to run all commands in Docker containers instead of your local machine environment (especially Apple chips have known issues).

   .. code-block::

      # Get <image-name> from .circleci/VERSIONS
      docker pull <image-name>

      # Run the docker container and land on interactive terminal
      docker run -it --name <container-name> <image-name>

      # Save Container State after running commands
      docker commit <container-name> <new-image-name>

      # Run the Saved Container next time
      docker run -it --name <new-container-name> <new-image-name>

#. ``pipenv`` quick-introduction

   - ``pipenv`` is a tool that aims to bring the best of all packaging worlds (bundler, composer, npm, cargo, yarn, etc.) to the Python world.
   - ``pipenv`` automatically creates and manages a virtualenv for your projects, as well as adds/removes packages from your ``Pipfile`` as you install/uninstall packages.
   - It also generates the ever-important ``Pipfile.lock``, which is used to produce deterministic builds.
   - ``pipenv shell`` can be used to enter the virtual environment
     - Alterntively: ``pipenv run`` can be used to run any Python commands in the virtual environment
     - For example:

       .. code-block::

         # Run pytest in pipenv virtual environment
         pipenv run pytest

         # Run pre-commit in pipenv virtual environment
         pipenv run pre-commit run --all-files

         # OR enter pipenv shell and run commands
         pipenv shell
         pytest
         pre-commit run --all-files

#. Create Python virtual environment using ``pipenv``

   .. code-block::
   
      pip install pipenv
      pipenv --python <VERSION>  # Creates a virtual environment for the project with specified VERSION; e.g. pipenv --python 3.9

#. Install and set-up Required Python Packages in editable mode

   .. code-block::

     pipenv run pip install -e .
     pipenv --help  # For help with all "pipenv" commands

     # For dev environment
     pipenv install

     # For production environment
     pipenv sync

     # You can provide specific categories defined in "Pipfile" if you wish
     pipenv install --categories=packages,build-packages,dev-packages,docs-packages,tests-packages
     pipenv sync --categories=default,build-packages,dev-packages,docs-packages,tests-packages
     pipenv --help

#. Initialize Pre-commit

   .. code-block::

     pipenv run pre-commit install

#. To run any Python commands, you should either be in ``pipenv`` shell (``pipenv shell`` to enter) or use ``pipenv run`` in front of the command

   .. code-block::

     # Example to run pytest
     pipenv run pytest

     # OR
     pipenv shell
     pytest

#. If you update ``setup.cfg``, ``pyproject.toml``, or ``Pipfile``

   - ``Pipfile.lock`` and ``requirements`` files would need to be regenrated. For this, you would need to have ``docker`` installed on your machine.

     .. code-block::

        pipenv run pre-commit run --all-files --hook-stage manual pipenv-lock

Important Python Versions
*************************

Python versions are defined in these places:

- ``pyproject.toml``
   Defines the python-version requirement of the project
- ``Pipfile``
   Defines python-version used to configure ``pipenv``
- ``.circleci/config.yml``
   Python-version used in CI flow

**NOTE**: When updating ``python`` version for; ensure that all ``pyproject.toml``, ``Pipfile``, and ``.circleci/config.yml`` are in sync.

Naming Conventions
******************

#. git branch naming convention

   - ``<username>/<feature/bugfix/hotfix>/<a-short-and-clear-description>``

   - e.g. ``john/feature/json-tests-should-support-iommu``

Code Conventions
****************

```generative_data_prep`` follows standard `PEP8 <https://peps.python.org/pep-0008/>`_ coding conventions.

Docstrings
**********

``generative_data_prep`` uses `Google style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ for formatting docstrings.

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
