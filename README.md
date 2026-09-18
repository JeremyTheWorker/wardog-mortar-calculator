# War Dog Mortar Distance Calculator

This Windows utility calculates the direct distance from a saved TEAM mortar
coordinate to the map coordinates under your mouse. It does not click, press
keys, read game memory, or aim the mortar.

The screen regions are calibrated for the 1920x1080 layout shown in
`206376~1.JPG`. Coordinates may be anywhere on the map. One coordinate unit is
treated as 100 meters.

![TEAM mortar and map-target coordinate example](206376~1.JPG)

## Run it

1. Double-click `run_mortar_calculator.bat`.
2. Wait until the window says `Mortar calculator ready`.
3. Mark the mortar location in-game so a line such as
   `TEAM x98.59, y109.82` appears in the top-left.
4. While that TEAM line is visible, press **F7**. The overlay will confirm
   `MORTAR SAVED`.
5. Open the map and point at the location you want to hit. Keep the pointer
   still so the map's `x##.##` and `y##.##` labels are visible.
6. Press **F8**.

A small overlay will show `RANGE ### m` for a few seconds. The console also
prints the mortar and target coordinates. You can press F8 for more targets;
press F7 again whenever the mortar moves. Press **Ctrl+F8** to stop.

The `206376~1.JPG` screenshot gives:

```text
Mortar: X=98.59, Y=109.82
Target: X=98.17, Y=109.88
Range: 42 m
```

## Manual setup

Python 3.12 or newer is recommended. From PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe mortar_calculator.py
```

## Check a saved screenshot

This mode is useful if live detection reports an error:

```powershell
.\.venv\Scripts\python.exe mortar_calculator.py `
  --image "206376~1.JPG" `
  --debug-output mortar_debug.jpg
```

The saved screenshot must show both the TEAM mortar line and the target X/Y
label. The debug image marks the two screen regions used for OCR.

## Troubleshooting

- **Could not read a TEAM mortar coordinate:** Create the TEAM coordinate line,
  wait until it is visible in the top-left, then press F7.
- **No mortar position is saved:** Press F7 successfully before using F8.
- **Could not read target X/Y:** Keep the full map open and let both pointer
  coordinate labels appear before pressing F8.
- **Overlay is hidden behind the game:** Use borderless-windowed display mode.
  The calculated range is still printed in the console.
- **Different resolution or UI layout:** The OCR regions will need to be
  recalibrated.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
