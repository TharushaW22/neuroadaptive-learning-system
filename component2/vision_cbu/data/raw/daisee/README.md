# DAiSEE dataset placeholder

This folder is intentionally empty until you obtain the dataset. The software, the dashboard
and all automated tests run without it.

Official dataset page (access, licence and citation requirements):
https://people.iith.ac.in/vineethnb/resources/daisee/

**Obtaining DAiSEE and complying with its licence is your responsibility.** Nothing in this
project downloads the dataset automatically.

## Where files go

```
data/raw/daisee/
  videos/          <- the video folder from the release (e.g. its "DataSet" folder)
  annotations/     <- the official label CSV files (e.g. "Labels" folder contents)
  README.md        <- this file
```

### Videos -> `data/raw/daisee/videos/`

Copy the released video folder **without restructuring it**. The expected official layout is

```
videos/DataSet/Train/<subjectID>/<clipID>/<clipID>.avi
videos/DataSet/Validation/<subjectID>/<clipID>/<clipID>.avi
videos/DataSet/Test/<subjectID>/<clipID>/<clipID>.avi
```

The audit derives the **participant ID from the `<subjectID>` folder** and the split from the
`Train/Validation/Test` folder. If your copy has a different layout, the audit reports
"participant id not derivable from folder layout" instead of guessing.

### Annotations -> `data/raw/daisee/annotations/`

Copy the official label CSVs unchanged, for example `TrainLabels.csv`, `ValidationLabels.csv`,
`TestLabels.csv` (and optionally `AllLabels.csv`). The split is inferred from each file name;
`AllLabels.csv` is treated as a union file without split information. Column names are
discovered at audit time (e.g. `ClipID`, `Boredom`, `Engagement`, `Confusion`, `Frustration`,
including stray whitespace) - the audit report shows exactly which column was matched.

## Rules

1. **Preserve the official directory structure and the official train/validation/test split files.**
2. **Do not rename clips or edit labels.** If a change is unavoidable, record what, why and how
   in `docs/dataset_guide.md` ("Local modifications log") and keep the originals.
3. **Do not commit videos or labels to Git.** `.gitignore` already excludes these folders and
   all video file types. Do not upload them to shared drives that the licence does not allow.
4. Keep the dataset on an encrypted / access-controlled disk if your institution requires it.

## After copying

```powershell
python scripts\audit_dataset.py           # inspect files, labels, splits, leakage
python scripts\extract_features.py        # cache per-frame features to data/processed/features
```
