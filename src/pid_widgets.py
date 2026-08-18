"""Custom Tkinter widgets for PIDTuner."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Tooltip:
    """A minimal hover tooltip for any Tk widget (Tkinter has no built-in
    one). Shows a small popup near the widget after a short delay, hides on
    leave/click. Keep a reference to the returned instance for the tooltip's
    bindings to stay alive as long as the widget does."""

    def __init__(self, widget, text, delay=400, wraplength=260):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel_pending()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel_pending(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.widget.winfo_viewable():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=self.text, justify="left",
                 background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=self.wraplength, padx=6, pady=4,
                 font=("TkDefaultFont", 8)).pack()

    def _hide(self, _event=None):
        self._cancel_pending()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def add_tooltip(widget, text, **kwargs):
    """Attach a hover tooltip to any widget. Returns the Tooltip instance —
    hang onto it if the widget outlives the immediate calling scope and you
    want to update/remove the tooltip later; otherwise it's fine to discard
    (the widget's own event bindings keep it alive)."""
    return Tooltip(widget, text, **kwargs)


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
