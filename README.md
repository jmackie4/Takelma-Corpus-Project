Hi! If you're reading this then that means you're interested in using the Takelma Corpus Project (TCP) program! This is my first time writing a README file, so I don't really know how you're supposed to structure these, so I'm just going to discuss the project here and what you need to get started!!! If you have any questions about using the project or if anything is unclear, feel free to email me at jmackie4@asu.edu!!!

What Even is TCP???
TCP is a program that processes a corpus of parallel texts and allows users to do a few things. These things are:
1. Print specific texts from the corpus
2. Find specific sequences of words within the corpus
3. Create an n-gram language model and generate text using it
4. Create a word aligner and provide initial glosses for a pair of parallel sentences

What Do I Need to Get Started???
In order to use TCP, you'll need to make sure the corpus of parallel texts meets the following requirements:
1. One of the languages must be English! If you're seeing this requirement, then that means I haven't modified TCP to be usable with any language pair!
2. The corpus of parallel texts you want to use with TCP must be a directory (folder) with two sub-directories in it. In addition, the two sub-directories must meet some requirements:
4. The sub-directories must contain the same number of plaintext files, and the plaintext files must have the same names between the two sub-directories
5. The files within a sub-directory must all be written in the same language
6. A regular expression pattern to give TCP that'll tokenize the source language!

And that's all when it comes to making sure your corpus of parallel texts can be processed correctly by TCP! 

There are also some libraries that you'll need to make sure you've installed:
1. nltk
2. spaCy (You'll also need to make sure you've downloaded the eng_core_web_sm language model from spaCy too!)
3. pandas
4. numpy


How to use TCP:
1. In the terminal or command line (whatever you want to call it) type python3 'your/path/to/TCP/main.py' and then press enter!
2. Then you just want to follow the instructions and provide TCP with the path to your corpus folder
3. Next it'll ask you to provide the names of the two sub-directories that hold the texts in your corpus. Make sure to type them exactly how they appear on your computer and separate them with a commma (e.g - Folder 1,Folder 2)
4. Keep following the interactive prompts! For any prompt after the 3rd one and before the main menu, utilize either 0 or 1 as your input!
5. Once you get to the main menu, then you can just type in the command you want TCP to conduct. Make sure to type it exactly how you see it on the menu!
6. In order to stop using TCP, make sure you're on the main menu and then type 'exit', and viola! You'll quit the program!


   
