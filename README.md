# PID Tuner

A desktop PID-tuning tool (ENGR105). Nine tuning methods, closed-loop
simulation, and a side-by-side method comparison (heatmap + radar).

Developer/build docs live in [`src/README.md`](src/README.md).

## Download

Grab the latest build from the [Releases](../../releases) page:
`PIDTuner-windows.zip`, `PIDTuner-macos.zip`, or `PIDTuner-linux.zip`.

## Running it (first-launch notes)

The app is unsigned, so each OS needs a one-time nudge to run it.

### Windows
Unzip, then double-click `PIDTuner.exe`. SmartScreen will warn:
**More info → Run anyway**. (It's a one-folder build — keep `PIDTuner.exe`
next to its `_internal` folder.)

### macOS
Gatekeeper blocks unsigned apps. Strip the quarantine flag once, then open:

    xattr -dr com.apple.quarantine "/path/to/PIDTuner.app"

(Or: try to open it, let it get blocked, then **System Settings →
Privacy & Security → Open Anyway**.)

### Linux
Unzip and run `./PIDTuner/PIDTuner`. The binary needs GLIBC ≥ 2.35
(Ubuntu 22.04+ / Debian 12+). Check yours with `ldd --version`; if it's
older, build from source instead (see the developer README).

## Build from source
See [`src/README.md`](src/README.md).