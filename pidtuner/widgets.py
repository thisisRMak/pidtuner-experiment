"""Custom Tkinter widgets for PIDTuner."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ClosableNotebook(ttk.Notebook):
    """A ttk.Notebook with closing ('X') buttons on each tab."""
    _style_ready = False
    _imgs = {}

    def __init__(self, master=None, on_closed=None, **kw):
        if not ClosableNotebook._style_ready:
            self._build_style()
            ClosableNotebook._style_ready = True
        kw["style"] = "Closable.TNotebook"
        super().__init__(master, **kw)
        self._on_closed = on_closed
        self._pressed_index = None
        self.bind("<ButtonPress-1>", self._on_press, True)
        self.bind("<ButtonRelease-1>", self._on_release, True)

    @classmethod
    def _make_x(cls, name, color):
        img = tk.PhotoImage(name, width=14, height=14)
        img.blank()  # fully transparent
        for i in range(3, 11):
            for w in (-1, 0, 1):
                img.put(color, (i + w, i))
                img.put(color, (i + w, 13 - i))
        cls._imgs[name] = img  # keep a reference alive
        return img

    @classmethod
    def _build_style(cls):
        cls._make_x("ctab_close", "#888888")
        cls._make_x("ctab_close_active", "#d62728")
        style = ttk.Style()
        try:
            style.element_create(
                "ctab.close", "image", "ctab_close",
                ("active", "ctab_close_active"), border=6, sticky="")
        except tk.TclError:
            pass  # already created in a prior instance
        style.layout("Closable.TNotebook", [
            ("Closable.TNotebook.client", {"sticky": "nswe"})])
        style.layout("Closable.TNotebook.Tab", [
            ("Closable.TNotebook.tab", {"sticky": "nswe", "children": [
                ("Closable.TNotebook.padding", {
                    "side": "top", "sticky": "nswe", "children": [
                        ("Closable.TNotebook.label", {"side": "left", "sticky": ""}),
                        ("ctab.close", {"side": "left", "sticky": ""}),
                    ]})
            ]})
        ])

    def _on_press(self, event):
        elem = self.identify(event.x, event.y)
        if "close" in elem:
            try:
                self._pressed_index = self.index("@%d,%d" % (event.x, event.y))
            except tk.TclError:
                self._pressed_index = None
            self.state(["pressed"])
            return "break"

    def _on_release(self, event):
        if not self.instate(["pressed"]):
            return
        elem = self.identify(event.x, event.y)
        try:
            index = self.index("@%d,%d" % (event.x, event.y))
        except tk.TclError:
            index = None
        if "close" in elem and index is not None and index == self._pressed_index:
            self.forget(index)
            if self._on_closed:
                self._on_closed()
        self.state(["!pressed"])
        self._pressed_index = None
