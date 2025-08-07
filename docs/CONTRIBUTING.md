# Contributing

We welcome contributions in the form of bug reports, bug fixes, improvements to the documentation,
ideas for enhancements (or the enhancements themselves!).

You can find a [list of current issues](https://github.com/NREL/electrolyzer/issues) in the project's
GitHub repo. Feel free to tackle any existing bugs or enhancement ideas by submitting a
[pull request](https://github.com/NREL/electrolyzer/pulls).

## Bug Reports

* Please include a short (but detailed) Python snippet or explanation for reproducing the problem.
  Attach or include a link to any input files that will be needed to reproduce the error.
* Explain the behavior you expected, and how what you got differed.

## Pull Requests

* Please reference relevant GitHub issues in your commit message using `GH123` or `#123`.
* Changes should be [PEP8](http://www.python.org/dev/peps/pep-0008/) compatible.
* Keep style fixes to a separate commit to make your pull request more readable.
* Docstrings are required and should follow the
  [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).
* When you start working on a pull request, start by creating a new branch pointing at the latest
  commit on [main](https://github.com/NREL/electrolyzer).
* The electrolyzer copyright policy is detailed in the [`LICENSE`](https://github.com/NREL/electrolyzer/blob/main/LICENSE).

## Documentation

When contributing new features, or fixing existing capabilities, be sure to add and/or update the
docstrings as needed to ensure the documentation site stays up to date with the latest changes.

To build the documentation locally, the following command can be run in your terminal in the docs
directory of the repository:

```bash
sh build_book.sh
```

In addition to generating the documentation, be sure to check the results by opening the following
path in your browser: `file://<path-to-electrolyzer>/electrolyzer/docs/_build/html/index.html`.

```{note}
If the browser appears to be out of date from what you expected to built, please try the following, roughly in order:
1. Reload the page a few times
2. Clear your browser's cache and open the page again.
3. Delete the `_build` folder and rebuild the docs
```

## Tests

The test suite can be run using `pytest .`

When you push to your fork, or open a PR, your tests will be run against the
[Continuous Integration (CI)](https://github.com/NREL/electrolyzer/actions) suite. This will start a build
that runs all tests on your branch against multiple Python versions, and will also test
documentation builds.