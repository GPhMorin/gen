# Gen
A library for analyzing pedigree data in Python

## Authors
Gilles-Philippe Morin & Jacob Côté, 2023.

## Purpose
So far, this library can:
* Parse an ASC pedigree file;
* Extract the probands, the parents, and the founders;
* Find all the known ancestors of a given individual;
* Intersect all known common ancestors of a set of individuals;
* Find all known most-recent common ancestors (MRCAs) of two individuals;
* Find the shortest distances from pairs of individuals to their MRCAs;
* Extract one member per family (in case you don't want siblings);
* Compute the inbreeding coefficient of a given individual.

## What This Cannot Do
* Compute kinship coefficients;
* Write a cool scientific article for you;
* Bring back the dead;
* And lots more things!

## How to Use
Simply `git clone` this project in the same folder as your Python script, and `from gen.gen import Gen`. Load a pedigree file using `Gen('path/to/pedigree.asc')`.

Tested with Python 3.10 and over.
