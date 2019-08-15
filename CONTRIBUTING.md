# Contributing

## Prove that What You've Created is Better than What Already Exists (or not)

It is paramount you prove that what you have is better than what the library looks like right now. This will not only have the functionality of being a source of *metadocumentation* but also a huge help for the eventual *reviewer(s)* of your *Pull Request*.

### But how exactly do you do that?

My suggestion is for you to create a script where you compare the existing approach to what you've come up with. This script will go into a `benchmarks` folder on the top level of the library. The `benchmarks` folder might not be merged into the `master` branch, however, it might play an important role in the `dev` branch.

## Version Control

Except in some cases &mdash; like better Documentation &mdash; this library uses [Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow), i.e., most content will first be buffered inside a `dev` or `feature` branch before being merged into the `master` branch.

However, I would advise you not to use `feature` indiscriminately, but rather choose a more appropriate name for your branch. For example, if you're contributing to a bug fix, I would suggest you use the format `bug_fix/<more_specific_name>`. In the end, the major contribution branches should look like:

- `feature`
- `code_improvement`
- `bug_fix`
- `docs`
