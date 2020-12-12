# Writer Identifier
Writer identification system (written in Python) for identifying the writer of a handwritten paragraph image.

## Introduction
For centuries, handwriting analysis has been a field of interest for numerous graphologists, psychologists, and forensic experts. With the advances in technologies and artificial intelligence, researchers from all over the world have dedicated their time and energy in automating the process of handwriting analysis using computers to ensure high efficiency and fast computations.

Writer identification is a sub-field of handwriting analysis that generally refers to the process of identifying the writer of a piece handwritten text.

In this repository, we present a writer identification system, where the system is required to identify the writer identity of a handwritten paragraph after getting trained upon handwriting of some different writers.

## Brief Description

In our work, we used texture-based approach for writer identification, we used the potential texture descriptor, namely Local Binary Pattern **(LBP)**, and we used Support Vector Machine **(SVM)** as a classifier to distinguish between feature vectors of different writers. We tested our system on _IAM handwritten dataset_, and we were able to achieve correct identification accuracy of about _**99%**_.

More information about the system architecture and pipeline is available [here](https://github.com/OmarBazaraa/WriterIdentifier/blob/master/docs/description.pdf).

## How To Use
1. Install Python 3 interpreter
2. Clone the repository
   ```Console
   git clone https://github.com/OmarBazaraa/WriterIdentifier.git
   ```
3. Install project dependencies
   ```Console
   pip install -r requirements.txt
   ```
4. Add testcases into `/data/testcases/` folder by following [this](/data/testcases/README.md) format
4. Run the project
   ```Console
   python ./src/main.py
   ```

**Note:** the system is tested on Windows using `Python 3.8.6` but should work on other platforms as well.

## Dataset Used
IAM offline handwriting database.

> U. Marti and H. Bunke.
> The IAM-database: An English Sentence Database for Off-line Handwriting Recognition.
> Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.
