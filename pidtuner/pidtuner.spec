# PyInstaller spec for the PID Tuner GUI.
#
# Build locally:   pyinstaller pidtuner.spec
# Output:          dist/PIDTuner       (one-folder)  or
#                  dist/PIDTuner.exe / dist/PIDTuner.app  depending on OS.
#
# The two non-obvious needs for this app:
#   1. matplotlib selects its TkAgg backend at runtime via matplotlib.use(),
#      so PyInstaller's static analysis can miss it — we force it in
#      hiddenimports and collect matplotlib's data files.
#   2. tkinter is pulled in automatically by PyInstaller's hooks, but on Linux
#      the build machine must actually have Tk installed (see the CI workflow).

import sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = collect_data_files("matplotlib")

hiddenimports = [
    "matplotlib.backends.backend_tkagg",
    "scipy.special._cdflib",          # occasionally missed by the scipy hook
    "PIL._tkinter_finder",            # lets Pillow load Tk image icons in the
                                      # matplotlib navigation toolbar when frozen
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5", "PySide2", "PySide6", "PyQt6"],  # trim Qt we don't use
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PIDTuner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI app — no console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PIDTuner",
)

# On macOS, also wrap the one-folder build into a .app bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PIDTuner.app",
        icon=None,
        bundle_identifier="org.engr105.pidtuner",
        info_plist={"NSHighResolutionCapable": "True"},
    )
