"""
neurobox.gui.mta_browser.preferences
======================================
User preferences dialog and persistence.

Currently exposes:

* **Naming conventions** — which :class:`NamingConfig`s the project
  scanner uses, and in what order.  Defaults to neurobox first,
  labbox-mta second.

Settings are persisted via :class:`PySide6.QtCore.QSettings` so they
survive across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore    import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from .data_layer import (
    NamingConfig, default_naming_configs, labbox_mta_naming,
    neurobox_naming,
)


__all__ = [
    "Preferences",
    "PreferencesDialog",
    "load_preferences",
    "save_preferences",
]


_SETTINGS_ORG = "neurobox"
_SETTINGS_APP = "mta_browser"
_KEY_ENABLED_NAMINGS = "scan/enabled_namings"     # ordered list of names
_KEY_NAMING_PRESETS  = {                           # name → preset
    neurobox_naming.name:   neurobox_naming,
    labbox_mta_naming.name: labbox_mta_naming,
}


# ─────────────────────────────────────────────────────────────────────── #
# Persisted preferences                                                       #
# ─────────────────────────────────────────────────────────────────────── #

@dataclass
class Preferences:
    """User preferences that affect browser behaviour.

    Attributes
    ----------
    enabled_namings :
        Ordered list of :class:`NamingConfig`s the project scanner
        uses.  The first config whose ``session_pattern`` matches a
        directory wins for that directory.  Empty list means "use the
        defaults" — i.e. :func:`default_naming_configs`.
    """
    enabled_namings: list[NamingConfig]


def load_preferences() -> Preferences:
    """Read preferences from QSettings.

    Returns the defaults (with both presets enabled, neurobox first)
    if no settings are stored or the stored values reference unknown
    presets.
    """
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    raw = s.value(_KEY_ENABLED_NAMINGS, None)
    if not raw:
        return Preferences(enabled_namings=default_naming_configs())
    if isinstance(raw, str):
        names = [n.strip() for n in raw.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in raw if str(n).strip()]
    out: list[NamingConfig] = []
    for n in names:
        cfg = _KEY_NAMING_PRESETS.get(n)
        if cfg is not None and cfg not in out:
            out.append(cfg)
    if not out:
        return Preferences(enabled_namings=default_naming_configs())
    return Preferences(enabled_namings=out)


def save_preferences(prefs: Preferences) -> None:
    """Persist *prefs* to QSettings."""
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    names = [c.name for c in prefs.enabled_namings]
    s.setValue(_KEY_ENABLED_NAMINGS, ",".join(names))
    s.sync()


# ─────────────────────────────────────────────────────────────────────── #
# Dialog                                                                      #
# ─────────────────────────────────────────────────────────────────────── #

class PreferencesDialog(QDialog):
    """Modal dialog for editing :class:`Preferences`.

    Shows a re-orderable list of available naming presets.  Each
    preset can be enabled/disabled, and the order in which they're
    applied during project scanning is editable via Up/Down buttons.

    Use as::

        dlg = PreferencesDialog(parent=self, prefs=load_preferences())
        if dlg.exec() == QDialog.Accepted:
            new_prefs = dlg.preferences()
            save_preferences(new_prefs)
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        prefs:  Optional[Preferences] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences — Naming Conventions")
        self.resize(520, 380)
        self._prefs = prefs or load_preferences()
        self._build_ui()
        self._populate()

    # ── UI construction ─────────────────────────────────────────── #

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        intro = QLabel(
            "Naming conventions used when scanning a project root.\n"
            "Checked entries are applied in order, top-to-bottom.\n"
            "The first convention whose session-name pattern matches\n"
            "a directory wins for that directory."
        )
        intro.setStyleSheet("color: #555;")
        outer.addWidget(intro)

        # List + up/down buttons
        body = QHBoxLayout()
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        body.addWidget(self._list, stretch=1)

        btn_col = QVBoxLayout()
        self._up_btn   = QPushButton("↑ Move up")
        self._down_btn = QPushButton("↓ Move down")
        self._reset_btn = QPushButton("Restore defaults")
        for b in (self._up_btn, self._down_btn, self._reset_btn):
            b.setMinimumWidth(120)
            btn_col.addWidget(b)
        btn_col.addStretch(1)
        self._up_btn.clicked.connect(lambda: self._move(-1))
        self._down_btn.clicked.connect(lambda: self._move(+1))
        self._reset_btn.clicked.connect(self._reset)
        body.addLayout(btn_col)
        outer.addLayout(body, stretch=1)

        # Per-entry detail (read-only) below the list
        self._detail = QLabel("")
        self._detail.setStyleSheet(
            "color: #444; padding: 4px; "
            "background: #f3f3f3; border-radius: 3px;"
        )
        self._detail.setWordWrap(True)
        outer.addWidget(self._detail)
        self._list.currentItemChanged.connect(
            lambda *_args: self._update_detail()
        )

        # OK / Cancel
        bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    # ── Population & state mapping ──────────────────────────────── #

    def _populate(self) -> None:
        """Fill the list from current ``self._prefs`` plus any unused
        presets at the bottom (unchecked)."""
        enabled_names  = [c.name for c in self._prefs.enabled_namings]
        self._list.clear()

        # Enabled presets first, in order
        for cfg in self._prefs.enabled_namings:
            self._add_item(cfg, checked=True)
        # Then any presets not currently enabled, unchecked
        for name, cfg in _KEY_NAMING_PRESETS.items():
            if name not in enabled_names:
                self._add_item(cfg, checked=False)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _add_item(self, cfg: NamingConfig, *, checked: bool) -> None:
        item = QListWidgetItem(cfg.name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        item.setData(Qt.UserRole, cfg)
        self._list.addItem(item)

    def _update_detail(self) -> None:
        item = self._list.currentItem()
        if not item:
            self._detail.setText("")
            return
        cfg: NamingConfig = item.data(Qt.UserRole)
        self._detail.setText(
            f"Pattern: {cfg.session_pattern.pattern}\n"
            f"Session marker glob: {cfg.session_marker_glob}\n"
            f"Trial   marker glob: {cfg.trial_marker_glob}"
        )

    # ── Actions ─────────────────────────────────────────────────── #

    def _move(self, delta: int) -> None:
        row = self._list.currentRow()
        new_row = row + delta
        if row < 0 or not (0 <= new_row < self._list.count()):
            return
        item = self._list.takeItem(row)
        self._list.insertItem(new_row, item)
        self._list.setCurrentRow(new_row)

    def _reset(self) -> None:
        self._prefs = Preferences(enabled_namings=default_naming_configs())
        self._populate()

    # ── Result accessor ─────────────────────────────────────────── #

    def preferences(self) -> Preferences:
        """Return the user's edited preferences (call after exec())."""
        out: list[NamingConfig] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.Checked:
                out.append(item.data(Qt.UserRole))
        if not out:
            # User unchecked everything — fall back to defaults rather
            # than persisting an empty config that would scan nothing.
            out = default_naming_configs()
        return Preferences(enabled_namings=out)
