# Dataset: Software Bug Reports
## Overview
The original dataset contains software bug reports collected from Mozilla, Eclipse, Open Office and NetBeans. It includes details such as bug descriptions, severity levels, and other metadata that can be useful for research and analysis in software engineering, bug triaging, and automated systems for bug report classification. The filtered datasets are from Mozilla, Eclipse and Open Office. They contain bug reports that have been filtered and processed to provide a unified severity scale across multiple projects. Certain bug reports with irrelevant severity labels were excluded, and the severity levels have been standardized to facilitate consistent evaluation.

## Source
The dataset is sourced from Kaggle and can be accessed through the following link: [Software Bug Reports - Kaggle Dataset](https://www.kaggle.com/datasets/samanthakumara/software-bug-reports)

## Dataset Details
Creator: Samantha Kumara
Platform: Kaggle
File Format: CSV files containing bug report metadata.
Content: The dataset includes bug reports with relevant features such as bug descriptions, severity levels, status, priority, and other useful information for classification and analysis.

## Data Processing and Filtering
Excluded Labels:
- N/A (Not Applicable): Bug reports with this label do not fall into a severity scale and have been removed.
- ’-’ (Default value): Bug reports marked with this default value were excluded due to their lack of meaningful severity classification.
- Enhancement: Bug reports marked as "Enhancement" were excluded, as they do not indicate a severity level but rather a feature request.

## Severity Scale Standardization
A key challenge with bug reports is the use of different severity scales across projects. To ensure consistency in analysis, we have unified the severity scale across the entire dataset.

- S1 → Blocker
- S2 → Major
- S3 → Normal
- S4 → Minor

## Final Severity Scale
After standardizing, the severity scale follows this order from most to least severe:

- Blocker
- Critical
- Major
- Normal
- Minor
- Trivial

## Purpose
The filtered datasets are now consistent in terms of severity labels, making it suitable for research focused on:

- Severity prediction
- Bug classification

## Licensing
No Licence from the original data set: [Software Bug Reports - Kaggle Dataset](https://www.kaggle.com/datasets/samanthakumara/software-bug-reports)