## This is a test of a tutorial to illustrate DVC Continuous Machine Learning.

*Step 1.* Please fork the repo.

*Step 2.* In line 19 of `code/train.py`, please change `learning_rate = 0.01`  to `learning_rate = 0.02` (or any other value of your choice).

*Step 3.* Please commit and submit a PR to this repo.

The goal is to check whether a user who is not a collaborator on this repository can initiate a Git Action to retrain the model.

**Open questions**
A) Will PRs from external users successfully trigger Git Actions? Will these Git Actions be able to access the necessary authorization credentials to pull from my data remote (a GDrive folder)?

B) Is it kosher to keep certain credentials (my GDrive ID & Secret) in the `.dvc/config` file, or will we need to locate another place to keep them for the purpose of this tutorial?
