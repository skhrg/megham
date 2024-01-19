# megham

A library for working with point clouds and related concepts.

## Introduction 
Over the last few years I have written a fair bit of code for both
the [Simons Observatory](https://simonsobservatory.org/) and [CLASS](https://sites.krieger.jhu.edu/class/)
that involves point clouds.
When writing that code I have found myself wanting for a cohesive library that handles all my point cloud
related tasks.
There are things that exist in bits and pieces
([numpy](https://numpy.org/), [scipy](https://scipy.org/), [sklearn's manifold module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold), [pycpd](https://github.com/siavashk/pycpd), etc.)
but none of them had all of the features I wanted or were implemented in ways that weren't ideal for my usecase.

Megham exists to help me bridge that gap.
Currently I am targeting only a small set of features that are relevant to my work:

* Fitting transforms between point clouds
* Point set registration (without known correspondence)
* Outlier detection
* Multi dimensional scaling

But other features may exist down the line (and PRs are welcome).

## Getting Started
To install this repository run:
```
pip install megham 
```
If you will be actively developing the code may want to instead clone this repository and run:
```
pip install -e .
```

Documentation can be found [here](https://skhrg.github.io/megham/)

## Contributing

All are welcome to contribute to this repository as long as long as the code is relevant.
In general contributions other than minor changes should follow the branch/fork -> PR -> merge workflow.
If you are going to contribute regularly, contact me to get push access to the repository.

### Style and Standards
In general contributions should be [PEP8](https://peps.python.org/pep-0008/) with commits in the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format.
This library follows [semantic versioning](https://semver.org/), so changes that bump the version should do so by editing `pyproject.toml`.

In order to make following these rules easier this repository is setup to work with [commitizen](https://commitizen-tools.github.io/commitizen/) and [pre-commit](https://pre-commit.com/).
It is recommended that you make use of these tools to save time.

Docstrings should follow the [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html). API reference docs are automatically built, but any additional narrative documentation or tutorials should go in the `docs` folder. This project uses [`mkdocs`](https://www.mkdocs.org/) to generate documentation.

#### Tool Setup
1. Install both tools with `pip install commitizen pre-commit`.
2. `cd` into the `megham` repository it you aren't already in it.
3. (Optional) Setup `commitizen` to automatically run when you run `git commit`. Follow instruction [here](https://commitizen-tools.github.io/commitizen/tutorials/auto_prepare_commit_message/).
4. Make sure the `pre-commit` hook is installed by running `pre-commit install`.

#### Example Workflow
1. Make a branch for the edits you want to make.
2. Code.
3. Commit your code with a [conventional commit message](https://www.conventionalcommits.org/en/v1.0.0/#summary). `cz c` gives you a wizard that will do this for you, if you followed Step 3 above then `git commit` will also do this (but not `git commit -m`).
4. Repeat steps 2 and 3 until the goal if your branch has been completed.
5. Put in a PR.
5. Once the PR is merged the repo version and tag will update [automatically](https://commitizen-tools.github.io/commitizen/tutorials/github_actions/).
