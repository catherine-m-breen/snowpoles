# About this fork
This repository is forked from ![catherine-m-breen/snowpoles](https://github.com/catherine-m-breen/snowpoles). Go there if you want a good, clear overview of what exactly this code does.

This fork has all of my improvements that haven't been merged into the official repository, including:

 - A tutorial for getting everything you need to work with this repository set up on Windows (see tutorial_windows.md)
 - No Miniconda/Anaconda needed. Ever.
 - Universal config file, so you can "set it and forget it" and not have to include flags every time you run a command
 - Dark mode
 - Progress-saving while labeling images with labeling.py, so you can pick up where you left off and not have to relabel hundreds of images
 - Using the images' unique filenames to adapt across directory structures
 - Full support of Python 3.13.3 and the newest versions of every library


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

Create your config file:
```
cp config-example.toml config.toml
```

Run the graphical configurator, which lets you pick file paths to use using your system's file dialog:
```
python configurator.py
```

If the above command failed, run the following command and retry:
```
pip install crossfiledialog
```

Just like that! Enjoy your improved experience.
