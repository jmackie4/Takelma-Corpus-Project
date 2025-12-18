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


   
