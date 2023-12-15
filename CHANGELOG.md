## 0.5.1 (2023-12-15)

### Fix

- don't rely on hardcoded epsilon

## 0.5.0 (2023-12-08)

### Feat

- add rigid cpd

## 0.4.0 (2023-11-10)

### Feat

- transforms have selectable basis
- added callback to cpd
- first pass of joint CPD algorhitm
- add function to apply affine transformation

### Fix

- linalg corrections and don't return tranform that increases err
- don't return last iter
- bugfixes galore, mostly indexing

## 0.3.0 (2023-10-06)

### Feat

- added smacof support in nonmetric edm and refactoried nonmetrix edm to match smacof's end condition
- add smacof and metric mds integration
- naive nonmetric mds implementation
- initialize to random coords if no guess is provided
- add metric mds
- add function to compute edm
- add classical mds

### Fix

- saner defaults
- off by one indexing bug

### Refactor

- switch from prints to logging

## 0.2.0 (2023-10-02)

### Feat

- add function to compute rigid transform

## 0.1.0 (2023-10-02)

### Feat

- Add functions for affine transformations
