# Contributing

## The General Flow of a Contribution

A "good" contribution would follow this flow:

1. (Sometimes Optional) Create an Issue.
1. Prove that what you've created is better than what already exists.
1. Create/modify an automated test to guarantee that what you did is not going to break some other thing inside the library.
1. Make sure your code follows this libraries editing conventions.
1. Create a *Pull Request* to an appropriate branch.

## First Things First: Create an Issue

Most topics on this library are far from trivial, newcomers might misunderstand some concepts and, thus, if they blindly try to create *Pull Request*, their efforts might be for naught.

Therefore, post your question on the [Issues Section](https://github.com/CamDavidsonPilon/lifetimes/issues) of the library first. It will be quickly (hopefully) labeled and other people's collaboration will provide enough feedback for you to know what to do next.

## Prove that What You've Created is Better than What Already Exists (or not)

It is paramount you prove that what you have is better than what the library looks like right now. This will not only have the functionality of being a source of *metadocumentation* but also a huge help for the eventual *reviewer(s)* of your *Pull Request*.

### But how exactly do you do that?

My suggestion is for you to create a script where you compare the existing approach to what you've come up with. This script will go into a `benchmarks` folder on the top level of the library. The `benchmarks` folder might not be merged into the `master` branch, however, it might play an important role in the `dev` branch.

This is very similar to what (Data) Scientists do when they create `Jupyter Notebooks`. In those, they expose their reasoning towards a solution, which is not intended for production, only to explain their thoughts.

## Tests

There are already quite a lot of tests in this library. However, nothing guarantees that what you're creating won't break an existing feature. It is recommended that you thus:

1. Go through all the existing tests.
    - Travis CI will do that automatically.
1. Examine the existing tests to see if they already guarantee that what you're doing is enough.
    - This can be difficult because you will probably not know all of the tests. Nevertheless, using `Ctrl + F` is always your friend. Anyway, try your best.
1. Write new tests *if* necessary.

Additionally, if it were me, even if there already exists a test covering my code, I might end up writing a custom one &mdash; or mentioning the name of the existing one &mdash; in my `benchmarks` file anyway, just for the sake of documentation.

## Editing Conventions

For the most part, this library follows [`PEP 8`](https://www.python.org/dev/peps/pep-0008/#a-foolish-consistency-is-the-hobgoblin-of-little-minds) conventions. Try to follow them when contributing. If you find code inside this library that does not respect those conventionse, please do create an issue and we will try to fix it. It's usually straight forward to fix it and it avoids a lot of pain in the long-term.

It is also crucial that you follow [`Numpy's Docstrings`](https://docs.scipy.org/doc/numpy/docs/howto_document.html) conventions when creating or editing `docstrings`. They are a subset of [`PEP 257`](https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings).

## Version Control

Except in some cases &mdash; like better Documentation &mdash; this library uses [Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow), i.e., most content will first be buffered inside a `dev` or `feature` branch before being merged into the `master` branch.

However, I would advise you not to use `feature` indiscriminately, but rather choose a more appropriate name for your branch. For example, if you're contributing to a bug fix, I would suggest you use the format `bug_fix/<more_specific_name>`. In the end, the major contribution branches should look like:

- `feature`
- `code_improvement`
- `bug_fix`
- `docs`
