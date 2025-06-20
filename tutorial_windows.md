# Installation and Usage on Windows

## Table of Contents
1. [Overview](#overview)
2. [Installing Python](#installing-python)
3. [Install VSCode](#install-vscode)
4. [Installing Git](#installing-git)
5. [Accessing the GitHub Repository](#accessing-the-github-repository)
6. [Preparing for First Use](#preparing-for-first-use)
7. [Running the Demo](#running-the-demo)

## Overview
The snowpoles repository expects some knowledge of python prior to running. However, we recognize that there are many researchers with active snowpole networks without necessarily a familiarity with github and python. This tutorial is to provide added context, particularly those with windows, to getting started with the model. 

## Installing Python
1. Open an internet browser and visit https://python.org/downloads/windows. Or, if the program you're reading this in supports it, you can click the link to go there.
2. Click the topmost link starting with "Latest Python 3 Release".
3. Scroll down to the *Files* section.
4. Click *Windows installer (64-bit)*.
5. If your browser asks you where you want to save the file, save it to your Downloads folder.
6. When the file finishes downloading, close your browser.
7. Open File Explorer (or your preferred file manager) and go to your Downloads folder.
8. Open the Python installer, which should be the most recent file. Its name starts with `python-`.
9. When the Python installer window opens, check *Add python.exe to PATH*. You may have to click the Python icon on the taskbar to get the installer window to show up.
10. Click *Install Now.*
11. When you see the *Setup was Successful* screen, close the Python installer. You can disable the path length limit if you'd like, but it won't affect anything here.
12. Delete the Python installer from your Downloads folder.


## Installing a code editor (VSCodium)
Note: In the paper, we used VSCode, the instructions below are for VSCodium. Both will work. For instructions to install VSCode on windows (or mac), please visit: https://code.visualstudio.com/download 
1. Open an internet browser and go to https://github.com/VSCodium/vscodium/releases. Or, if the program you're reading this in supports it, you can click the link to go there.
2. Under the *x86 64bits* header, click the link under *Windows > User Installer*. It should start with `VSCodiumUserSetup-x64`.
3. If your browser asks you where you want to save the file, save it to your Downloads folder.
4. When the file finishes downloading, close your browser.
5. Open File Explorer (or your preferred file manager) and go to your Downloads folder.
6. Open the VSCodium installer, which should be the most recent file. Its name starts with `VSCodiumUserSetup-x64`.
7. When the VSCodium installer window opens, click *I accept the agreement*. This accepts the MIT license, with no data collected. You may have to click the VSCodium icon on the taskbar to get the installer window to show up.
8. Click *Next*.
9. On the *Select Destination Location* screen, click *Next*.
10. On the *Select Start Menu Folder* screen, click *Next*.
11. On the *Select Additional Tasks* screen, check every box except the top one. If you want a desktop icon, check the top box, too.
12. Click *Install*.
13. When you reach the *Completing the VSCodium Setup Wizard* screen, uncheck *Launch VSCodium*.
14. Click *Finish*.
15. Delete the VSCodium installer from your Downloads folder.


## Installing Git
1. Open an internet browser and go to https://git-scm.com/downloads/win. Or, if the program you're reading this in supports it, you can click the link to go there.
2. Click *Git for Windows/x64 Setup.*
3. If your browser asks where to save the file, save it to your Downloads folder.
4. When the file finishes downloading, close your browser.
5. Open File Explorer (or your preferred file manager) and go to your Downloads folder.
6. Open the Git installer, which should be the most recent file. Its name starts with `Git-`.
7. Answer *Yes* to the User Account Control prompt.
8. When the Git installer window opens, click *Next*. You may have to click the Git icon on the taskbar to get the installer window to show up.
9.  On the *Select Destination Location* screen, click *Next*.
10. On the *Select Components* screen, click *Next*.
11. On the *Select Start Menu Folder* screen, click *Next*.
12. On the *Choosing the default editor used by Git* screen, select *Use VSCodium as Git's default editor* from the dropdown (if using VSCode, select VSCode as the default editor).
13. Click *Next*.
14. On every page from here forward, click *Next* without changing any settings.
15. When you reach the *Completing the Git Setup Wizard* screen, uncheck *Launch Git Bash*.
16. Click *Finish*.
17. Delete the Git installer from your Downloads folder.


## Accessing the GitHub repository
1. Launch VSCodium.
2. Press `Ctrl+Shift+P`.
3. Type `clone`.
4. Make sure *Git: Clone* is highlighted.
5. Press `Enter`.
6. Type or paste `https://github.com/catherine-m-breeen/snowpoles`.
7. You will see a file picker window. Click *New Folder*.
8. Type `GitHub`, and press `Enter` twice, leaving at least half a second between presses.
9. Click *Select as Repository Destination*.
10. On the *Would you like to open the cloned repository?* screen, click *Open*.
11. On the *Do you trust the authors of the files in this folder?* popup, click *Yes, I trust the authors*.


## Preparing for First Use
1. Press `` Ctrl+` ``. The `` ` `` key is to the left of the 1 on US keyboards. This opens the terminal.
2. Type `python -m venv venv` into the terminal and press `Enter`.
3. Type `venv/Scripts/pip install -r requirements.txt` into the terminal and press `Enter`. You may have to run the command more than once if the download fails due to a network issue. In that case, it'll pick up from after the last fully downloaded package.
4. Open a browser and go to https://aka.ms/vs/16/release/vc_redist.x64.exe.
5. If your browser asks where you want to save the file, save it to your Downloads folder.
6. When the file finishes downloading, close your browser.
7. Open File Explorer (or your preferred file manager) and go to your Downloads folder.
8. Open the file you downloaded. Its name is `VC_redist.x64.exe`.
9. When the installer opens, check *I agree to the license terms and conditions*.
10. Click *Install*.
11. Answer *Yes* to the User Account Control prompt.
12. When you reach the *Setup Successful* screen, close the installer window.
13. Delete `VC_redist.x64.exe` from your Downloads folder.


## Running the Demo
1. Type `venv/Scripts/python src/demo.py` into the terminal and press `Enter`.
2. If you get an error that starts with `OSError: [WinError 126]`, do steps 4-13 of **Preparing for First Use**, then re-run step 1.
3. When the script finishes, predictions will be stored in the `demo_predictions` folder.