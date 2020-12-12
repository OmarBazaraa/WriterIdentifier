# Writer Identifier
This directory contains IAM offline handwriting database.

This is an optional directory that was being used during development to generate testcases and analyze the behavior of the system.

## Dataset
Download the full IAM handwriting dataset from this [link](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

## Folder Structure
The folder is expected to be in the following structure:

```
dataset
├── forms
|   ├── image_1.png
|   ├── image_1.png
|   └── ...
|
└── meta.txt
```

- `forms` sub-folder contains handwriting images in any format (e.g. png, jpg, ..etc).
- `meta.txt` contains metadata about each of handwriting images.
