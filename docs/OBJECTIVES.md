# Objectives

| Field | MO-1 |
|---|---|
| ID | MO-1 |
| What is measured | Absolute error (mm) of the system's width and height versus a hand ruler/calliper measurement |
| How | measurement/ script compares system output with a hand-recorded CSV for each instance and prints a table, mean, median and worst case |
| Target | Mean absolute error 2 mm or less, worst case 5 mm or less, over 10 or more instances (object about 150 mm long, so 2 mm is about 1.3 percent) |
| Conditions | Phone camera, locked focus and zoom, 1200x1600, camera roughly overhead (tilt under 10 degrees), marker flat in object plane, indoor light |
| Not included | Tilt above 10 degrees, marker partly hidden, object touching the frame edge |

| Field | MO-2 |
|---|---|
| ID | MO-2 |
| What is measured | Mask IoU on the held-out test split |
| How | evaluation script on dataset/splits/test.txt only, mean IoU between predicted and labelled masks |
| Target | 0.90 or more |
| Conditions | Test split of 10 images, checkpoint chosen on val only, seed 42 |
| Not included | Images from the tilt experiment |

## Results
(raw outputs pasted here as they are produced)
