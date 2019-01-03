# Writer Identifier
Writer identification system (written in Python) for identifying the writer of a handwritten paragraph image.

## Introduction
For centuries, handwriting analysis has been a field of interest for numerous graphologists, psychologists, and forensic experts. With the advances in technologies and artificial intelligence, researchers from all over the world have dedicated their time and energy in automating the process of handwriting analysis using computers to ensure high efficiency and fast computations.

Writer identification is a sub-field of handwriting analysis that generally refers to the process of identifying the writer of a piece handwritten text.

In this repository, we present a writer identification system, where the system is required to identify the writer identity of a handwritten paragraph after getting trained upon handwriting of some different writers.

## Brief description

In our work, we used texture-based approach for writer identification, we used the potential texture descriptor, namely Local Binary Pattern **(LBP)**, and we used Support Vector Machine **(SVM)** as a classifier to distinguish between feature vectors of different writers. We tested our system on _IAM handwritten dataset_, and we were able to achieve correct identification accuracy of about _**99%**_.

More information about the system architecture and pipeline is available [here](https://github.com/OmarBazaraa/WriterIdentifier/blob/master/docs/description.pdf).

## How to use
1. Install Python 3 interpreter.
2. Clone this repository.
   ```Console
   git clone https://github.com/OmarBazaraa/WriterIdentifier.git
   ```
3. Install project dependencies.
   ```Console
   pip install -r requirements.txt
   ```
4. Add test cases into `/data/testcases/` folder.
4. Run the project.
   ```Console
   python ./src/main.py
   ```

## Test cases format
Each test case consists of two parts: training data and test data.

#### 1. Training data
Training images of a specific writer should be grouped together in folder named after that writer.

#### 2. Test data
Test images (those that our system is required to identify) should be in the root of each test case folder.

#### Example
Test cases format example

```
testcases
├── 001
|   ├── writer1
|   |   ├── training_image1.png
|   |   ├── training_image2.png
|   |   └── ...
|   |
|   ├── writer2
|   |   ├── training_image1.png
|   |   ├── training_image2.png
|   |   └── ...
|   |
|   ├── ...
|   |
|   ├── test_image1.png
|   ├── test_image1.png
|   └── ...
|
├── 002
|   └── ...
|
└── ...
```

#### Notes
* No specific format is required for naming the folders inside '/data/testcases/'.
* Images could be in any format (e.g. png, jpg, ..etc).
* No limitations on the number of test cases, writers, images per writer, or test images.
* The above test case folder structure should be followed for the system to run correctly.

# Database used
IAM offline handwriting database.

> U. Marti and H. Bunke.
> The IAM-database: An English Sentence Database for Off-line Handwriting Recognition.
> Int. Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.
