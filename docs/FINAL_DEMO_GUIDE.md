# Motor Health Monitoring — Final Demonstration Guide

## Launch the dashboard

In PowerShell:

```powershell
cd C:\Users\hp\Motor_Health_Monitoring
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.
Start on **Overview**. The normal demonstration uses files already saved in
`results`; do not press **Capture 30 seconds from ESP32 (COM8)** during the
offline demonstration.

## Recommended page-by-page sequence

### 1. Overview — introduce the system

Explain that the project reads vibration measurements from an ESP32 MPU6500,
extracts statistical features, and compares them with bearing-condition
patterns learned from the Paderborn bearing dataset.

Point out the limitation notice: the trained bearing model has not been
validated on this blender motor. The Overview also summarizes the independent
bearing test dataset when its CSV is available.

### 2. Model Evaluation — show bearing-dataset evaluation

This page displays the saved independent evaluation report and candidate-model
comparison. The current report is for the bearing test dataset: it records
30,788 test windows from six unseen bearings, with accuracy 0.9007 and macro
F1 0.9016.

Emphasize that these are results on the bearing dataset, not measurements of
blender accuracy. The report itself lists limitations, including that windows
from the same bearing are not independent observations.

### 3. Dataset Explorer — inspect the bearing test data

The page initially loads the saved independent test CSV. Show its preview and
the feature or condition-distribution charts. An uploaded feature CSV can be
selected instead.

### 4. Predict Condition — optional bearing-model example

This page accepts seven manually entered features and runs the existing model.
Use it only as an example of the bearing-model interface. Its output is
experimental and must not be presented as a blender diagnosis. Avoid suggesting
that the form defaults are measured sensor data.

### 5. Live Monitoring — show the saved ESP32 experiment offline

Without pressing the capture button, the page reads the latest saved recording
and feature window. It displays the latest saved accelerometer readings,
feature values, recording/window timestamps, the experimental bearing-model
class and model scores. It also provides downloads for the saved recording and
feature-window CSVs.

The current saved files contain 317 raw rows and three feature windows; the
saved capture timestamp shown by the app is based on the recording file time.
Call this **saved ESP32 sensor data**, not a live connection during this
demonstration. The latest experimental model output is not a validated blender
fault indication.

The sensor CSV has an `acs_adc` field whose saved values are `-1`. The app
labels these as unavailable; they are not current measurements.

### 6. Motor Comparison — compare recorded operating conditions

Use the separate **Motor Comparison** page in Streamlit's multipage
navigation. It compares the available motor OFF and motor ON feature summaries.
Point out differences in measured vibration features, not a fault conclusion:
mounting, orientation, noise, and operating conditions can all affect them.

The page has a comparison CSV download, a report viewer, and a report download.
Live Monitoring also provides downloads for the raw saved sensor CSV and
extracted feature-window CSV.

## Keep the results distinct

- **Bearing evaluation:** model metrics and class results against the
  Paderborn/independent bearing test data.
- **Blender experiment:** ESP32 vibration recordings and derived features from
  the blender. The model's displayed class is experimental because transfer
  from bearing data to this motor has not been validated.
- Do not say the blender is healthy or faulty based on the displayed class.
- Do not report `acs_adc = -1` as zero or measured current.
- No COM8 capture is needed to demonstrate saved sensor readings, features,
  predictions, comparisons, reports, or downloads.

## Likely faculty questions

**What did the model learn from?**  
Bearing recordings in the Paderborn dataset, with bearing-condition labels.

**Does the reported accuracy show how well it diagnoses the blender?**  
No. That accuracy is measured on the independent bearing test dataset and
does not establish performance on this blender.

**Why show an experimental prediction at all?**  
It demonstrates the end-to-end feature-to-model software path. It is a
research/demo output, not a validated diagnosis.

**What does `acs_adc = -1` mean?**  
Current sensing is unavailable or unvalidated in this setup. `-1` is a
sentinel, not a current reading.

**Is the demonstration reading COM8 right now?**  
No, unless a new capture was explicitly started. The offline demonstration
uses the timestamped saved CSV files.

**What would be needed to validate the model for the blender?**  
Appropriately collected and labeled blender data under controlled conditions,
followed by a separate evaluation. The current bearing results are not a
substitute for that validation.

## Safe procedure for a future live ESP32 test

1. Do not run the live capture as part of the offline demonstration.
2. Before a future test, follow lab/supervisor safety guidance. Keep clear of
   moving parts and do not adjust sensor wiring while equipment is energized.
3. Close Arduino Serial Monitor and stop every other process using COM8. Only
   one process can own the serial port at a time.
4. Connect the ESP32 on the configured COM8 port, then select **Live Monitoring**
   and press **Capture 30 seconds from ESP32 (COM8)** once. Wait for capture
   and window extraction to finish; the app will update the saved CSVs.
5. If COM8 is unavailable or another application has it open, stop and resolve
   the port conflict rather than starting another collector in parallel.
6. Describe resulting predictions as experimental bearing-model output. Do not
   treat them as proof of blender health or failure, and do not interpret
   `acs_adc = -1` as current.
