# Welcome to the Takelma Corpus Project!!

Hi! If you're reading this then that means you're interested in using the Takelma Corpus Project (TCP) program! This is my first time writing a README file, so I don't really know how you're supposed to structure these, so I'm just going to discuss the project here and what you need to get started!!! If you have any questions about using the project or if anything is unclear, feel free to email me at jmackie4@asu.edu!!!

## **What Even is TCP???**

TCP is a program that processes a corpus of parallel texts and allows users to do a few things. These things are:
1. Print specific texts from the corpus
2. Find specific sequences of words within the corpus
3. Create an n-gram language model and generate text using it
4. Create a word aligner and provide initial glosses for a pair of parallel sentences

## **What Do I Need to Get Started???**

In order to use TCP, you'll need to make sure the corpus of parallel texts meets the following requirements:
1. One of the languages must be English! If you're seeing this requirement, then that means I haven't modified TCP to be usable with any language pair!
2. The corpus of parallel texts you want to use with TCP must be a directory (folder) with two sub-directories in it. In addition, the two sub-directories must meet some requirements:
4. The sub-directories must contain the same number of plaintext files, and the plaintext files must have the same names between the two sub-directories
5. The files within a sub-directory must all be written in the same language
6. A regular expression pattern to give TCP that'll tokenize the source languages

Also make sure that you have all the necessary dependencies installed, you can find those in the requirements.txt file! 


## **How to use TCP:**
1. First just run the main script using the following in the terminal
```
python 'your/specific/path/Takelma Corpus Project'
```
2. You'll then see this prompt, insert the path to the directory that holds your corpus!
```angular2html
Please enter the main path that holds the corpus:
```
3. The next step will be to type in the names of the directories that hold the texts you want to use! Make sure you separate them by comma and type them exactly as they appear on the screen!
```
Please enter the names of the folders that hold the texts:
```
4. Next you'll need to provide TCP with your regex pattern so that it can tokenize the sentences in the non-English language!
```angular2html
Please enter your regex pattern:
```
5. Now you'll need to state what kind of n-gram model you want to create by giving a number! Try to keep it below 5 tho...
```angular2html
Please enter what kind of n-gram model you'd like to make by giving a number:
```
6. Then you need to set which language you want to use, please make sure to set this option to the non-English language!
```angular2html
Please choose which language you want to use using 0 or 1:
```
7. Now it's time to choose your aligner! Again, use a number to pick which aligner you want. For now, I suggest the entropy aligner!
```angular2html
Please enter your choice of aligner using the integer associated with the aligner: 
```

After all of that you'll finally be at the main menu! Here you'll want to just type out what you want to do given the options. Make sure you spell the options correctly, so try to just copy and paste the options!
```angular2html
0: get text
1: get titles
2: use n-gram model
3: find sequence
4: use aligner
Please enter what you want to do:
```
**Now if you want to exit out of TCP, just make sure you're on the main menu and type exit!!!**

Once you type out your option, just follow the interactive menu like how you did to setup TCP and enjoy! Once you do an action, you'll be sent back to the main menu!

   
