# About this fork
This repository is forked from ![catherine-m-breen/snowpoles](https://github.com/catherine-m-breen/snowpoles). Go there if you want a good, clear overview of what exactly this code does.

This fork has all of my improvements that haven't been merged into the official repository, including:

 - A tutorial for getting everything you need to work with this repository set up on Windows (see tutorial_windows.md)
 - Progress-saving while labeling images with labeling.py, so you can pick up where you left off and not have to relabel hundreds of images
 - Dark mode
 - No Miniconda/Anaconda needed. Ever.


# Switching from the official repository
Add this repository as a remote:
```
git remote add Nesitive https://github.com/Nesitive/snowpoles
```

Create a branch for the changes and switch to it:
```
git branch Nesitive
git checkout Nesitive
```

Pull in the changes:
```
git pull Nesitive main
```

Just like that! Enjoy your improved experience.