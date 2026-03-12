import sys
import numpy as np
import math
from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QScrollArea, QSlider,
    QSpinBox, QPushButton, QRubberBand, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QEvent, QTimer

_app = None
_windows = []

def get_application():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app

class DragScrollArea(QScrollArea):
    """Scrollable area that supports dragging with the mouse"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.viewport().setMouseTracking(True)
        self.setMouseTracking(True)
        self._dragging = False
        self._startPos = QPoint()
        self._hStart = 0
        self._vStart = 0

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self._startPos = ev.pos()
            self._hStart = self.horizontalScrollBar().value()
            self._vStart = self.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging:
            delta = ev.pos() - self._startPos
            self.horizontalScrollBar().setValue(self._hStart - delta.x())
            self.verticalScrollBar().setValue(self._vStart - delta.y())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

class ImageLabel(QLabel):
    """Label that shows image coordinates and pixel values on mouse hover plus a dynamic scale bar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        win = self.window()
        if not isinstance(win, ImageWindow) or win.scale_bar_params is None:
            return

        pix_size = win.scale_bar_params.get('pixel_size_um', None)
        if pix_size is None or pix_size <= 0:
            return

        vp = win.scroll.viewport()
        W, H = vp.width(), vp.height()
        f = 0.2  # ~20% of viewport width
        L_raw = (f * W) / max(win.zoom, 1e-9) * pix_size
        if L_raw <= 0:
            return

        # Build a "nice" list of lengths (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, ...)
        min_exp, max_exp = -3, 6
        factors = (1, 2, 5)
        nice_list = []
        for exponent in range(min_exp, max_exp + 1):
            for factor in factors:
                val = factor * (10 ** exponent)
                if val >= 0.05:
                    nice_list.append(val)
        nice_list.sort()

        L_snap = nice_list[0]
        for candidate in nice_list:
            if candidate <= L_raw:
                L_snap = candidate
            else:
                break

        bar_px = max(1, int(round((L_snap / pix_size) * win.zoom)))
        label_text = f"{L_snap:g} µm"

        x0 = int(W * 0.9) - bar_px
        thickness = max(1, H // 100)
        y0 = H - int(H * 0.04) - thickness

        h_off = win.scroll.horizontalScrollBar().value()
        v_off = win.scroll.verticalScrollBar().value()

        color = QColor('white' if win.scale_bar_params.get('color','white') == 'white' else 'black')
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawRect(h_off + x0, v_off + y0, bar_px, thickness)
        painter.setPen(color)
        painter.drawText(h_off + x0, v_off + y0 - 4, label_text)
        painter.end()

    def wheelEvent(self, ev):
        win = self.window()
        if not isinstance(win, ImageWindow):
            return super().wheelEvent(ev)

        steps = ev.angleDelta().y() / 120.0
        base  = 1.2

        old_zoom = win.zoom
        requested = old_zoom * (base ** steps)

        orig_h, orig_w = win.orig.shape[:2]
        vp = win.scroll.viewport().size()
        min_zoom_x = vp.width()  / max(orig_w, 1)
        min_zoom_y = vp.height() / max(orig_h, 1)
        dynamic_min = min(min_zoom_x, min_zoom_y)
        new_zoom = max(dynamic_min, min(requested, 20.0))

        win.zoom = new_zoom
        win._render()

        factor = new_zoom / max(old_zoom, 1e-9)
        hbar = win.scroll.horizontalScrollBar()
        vbar = win.scroll.verticalScrollBar()
        pos = win.scroll.viewport().mapFromGlobal(ev.globalPos())
        cx, cy = hbar.value() + pos.x(), vbar.value() + pos.y()
        hbar.setValue(int(cx*factor - pos.x()))
        vbar.setValue(int(cy*factor - pos.y()))
        ev.accept()

    def mouseMoveEvent(self, ev):
        win = self.window()
        if not isinstance(win, ImageWindow):
            super().mouseMoveEvent(ev)
            return

        if win.orig is None or win.orig.size == 0:
            win.info.setText("No valid image data")
            super().mouseMoveEvent(ev)
            return

        h, w = win.orig.shape[:2]
        ix = int(round(ev.x() / max(win.zoom, 1e-9)))
        iy = int(round(ev.y() / max(win.zoom, 1e-9)))
        if 0 <= ix < w and 0 <= iy < h:
            val = win.orig[iy, ix]
            if isinstance(val, (np.complexfloating, complex)):
                win.info.setText(f"X: {ix}, Y: {iy}, Mag: {abs(val):.3f}, Ph: {np.angle(val):.3f}")
            elif np.issubdtype(np.array(val).dtype, np.floating):
                win.info.setText(f"X: {ix}, Y: {iy}, Val: {float(val):.3f}")
            else:
                win.info.setText(f"X: {ix}, Y: {iy}, Val: {val}")
        else:
            win.info.setText(f"X: {ix}, Y: {iy} (out of bounds)")
        super().mouseMoveEvent(ev)

class ImageWindow(QWidget):
    """Main image viewer window with pixel-value display, dynamic scale bar, and a Save Snapshot button."""
    def __init__(self, arr, title="Image Viewer"):
        super().__init__()
        self.zoom = 1.0
        self.orig = None
        self.scale_bar_params = None
        self.setWindowTitle(title)
        self._build_ui()
        self.set_image(arr)

        # Initial sizing & centering
        screen = QApplication.primaryScreen().availableGeometry()
        screen_h = screen.height()
        screen_w = screen.width()
        initial_size = int(screen_h * 0.8)
        self.resize(initial_size, initial_size)
        self.move((screen_w - initial_size) // 2, (screen_h - initial_size) // 2)
        self.setMinimumSize(400, 400)

        # Auto-zoom to fit after widgets exist
        QTimer.singleShot(0, self._auto_fit_image)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        self.info = QLabel("X: -, Y: -, Val: -")
        self.info.setStyleSheet("QLabel { padding:5px; font-family:monospace; }")
        layout.addWidget(self.info)

        # Scroll area + image label
        self.scroll = DragScrollArea()
        self.label = ImageLabel()
        self.scroll.setWidget(self.label)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll)

        # Save Snapshot button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.save_btn = QPushButton("Save Snapshot")
        self.save_btn.clicked.connect(self.save_snapshot)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        # Window/level controls
        ctrl = QHBoxLayout()
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_spin = QSpinBox()
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_spin = QSpinBox()

        self.ww_spin.setMaximum(999999)
        self.wl_spin.setMinimum(-999999)
        self.wl_spin.setMaximum(999999)

        for lbl, pair in [("WW:", (self.ww_slider, self.ww_spin)), ("WL:", (self.wl_slider, self.wl_spin))]:
            ctrl.addWidget(QLabel(lbl))
            ctrl.addWidget(pair[0]); ctrl.addWidget(pair[1])
        layout.addLayout(ctrl)

        # Connect sliders and spinboxes
        self.ww_slider.valueChanged.connect(self._update_ww_from_slider)
        self.ww_spin.valueChanged.connect(self._update_ww_from_spin)
        self.wl_slider.valueChanged.connect(self._update_wl_from_slider)
        self.wl_spin.valueChanged.connect(self._update_wl_from_spin)

    # ---- WW/WL handlers (ImageWindow is the owner of these controls) ----
    def _update_ww_from_slider(self, value):
        if hasattr(self, 'data_range'):
            ww = (value / 10000.0) * self.data_range
            self.ww_spin.blockSignals(True)
            self.ww_spin.setValue(int(max(1, ww)))
            self.ww_spin.blockSignals(False)
            self._render()

    def _update_ww_from_spin(self, value):
        if hasattr(self, 'data_range') and self.data_range > 0:
            slider_val = int((value / self.data_range) * 10000)
            slider_val = max(1, min(10000, slider_val))
            self.ww_slider.blockSignals(True)
            self.ww_slider.setValue(slider_val)
            self.ww_slider.blockSignals(False)
            self._render()

    def _update_wl_from_slider(self, value):
        if hasattr(self, 'data_min'):
            wl = self.data_min + (value / 10000.0) * self.data_range
            self.wl_spin.blockSignals(True)
            self.wl_spin.setValue(int(wl))
            self.wl_spin.blockSignals(False)
            self._render()

    def _update_wl_from_spin(self, value):
        if hasattr(self, 'data_min') and self.data_range > 0:
            slider_val = int(((value - self.data_min) / self.data_range) * 10000)
            slider_val = max(0, min(10000, slider_val))
            self.wl_slider.blockSignals(True)
            self.wl_slider.setValue(slider_val)
            self.wl_slider.blockSignals(False)
            self._render()
    # --------------------------------------------------------------------

    def set_image(self, arr):
        if np.iscomplexobj(arr):
            self.orig = np.abs(arr).astype(np.float64)
        else:
            self.orig = arr.astype(np.float64)

        mn, mx = float(self.orig.min()), float(self.orig.max())
        if mn == mx:
            mn -= 0.5; mx += 0.5
        rng = mx - mn

        # slider ranges
        self.ww_slider.setRange(1, 10000)
        self.ww_slider.setValue(10000)
        self.ww_spin.setRange(1, 999999)
        self.ww_spin.setValue(int(rng))
        self.ww_spin.setSingleStep(max(1, int(rng/100)))

        self.wl_slider.setRange(0, 10000)
        self.wl_slider.setValue(5000)
        self.wl_spin.setRange(int(mn)-1000, int(mx)+1000)
        self.wl_spin.setValue(int((mn+mx)/2))
        self.wl_spin.setSingleStep(max(1, int(rng/100)))

        self.data_min = mn
        self.data_max = mx
        self.data_range = rng

        self._render()

    def add_scale_bar(self, params):
        self.scale_bar_params = {
            'length_um':     params.get('length_um', 10.0),
            'pixel_size_um': params.get('pixel_size_um', 1.0),
            'position':      params.get('position', 'bottom-right'),
            'color':         params.get('color', 'white'),
            'margin':        params.get('margin', 5),
            'thickness':     params.get('thickness', 2),
            'font_scale':    params.get('font_scale', 0.4)
        }
        self._render()

    def _render(self):
        if self.orig is None:
            return

        ww = max(0.001, float(self.ww_spin.value()))
        wl = float(self.wl_spin.value())

        lo, hi = wl - ww/2, wl + ww/2
        if hi <= lo:
            hi = lo + 0.001

        norm = np.clip((self.orig - lo)/(hi - lo), 0, 1)
        img8 = (norm * 255).astype(np.uint8)

        h, w = img8.shape[:2]
        arr8 = np.ascontiguousarray(img8)
        qimg = QImage(arr8.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        sw = max(1, int(pix.width() * self.zoom))
        sh = max(1, int(pix.height() * self.zoom))
        scaled = pix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.FastTransformation)

        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())
        self.scroll.viewport().update()

    def save_snapshot(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not filename:
            return
        if not filename.lower().endswith('.png'):
            filename += '.png'
        pixmap = self.scroll.viewport().grab()
        pixmap.save(filename, "PNG")

    def _auto_fit_image(self):
        if self.orig is None:
            return
        vp = self.scroll.viewport()
        vp_width = max(vp.width(), 1)
        vp_height = max(vp.height(), 1)
        img_height, img_width = self.orig.shape[:2]
        zoom_x = vp_width / max(img_width, 1)
        zoom_y = vp_height / max(img_height, 1)
        self.zoom = min(zoom_x, zoom_y)
        self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'orig') and self.orig is not None:
            self._auto_fit_image()

    def eventFilter(self, src, ev):
        return super().eventFilter(src, ev)

class ROISelector(QDialog):
    """ROI selector dialog with WW/WL sliders, dynamic scale bar, and Save Snapshot."""
    def __init__(self, img: np.ndarray, pixel_size_um: float = None):
        super().__init__()
        self.zoom = 1.0
        self.selected = QRect()
        self.has_selection = False
        self.pixel_size_um = pixel_size_um
        self.setWindowTitle("Draw ROI and click Confirm")
        self.setModal(True)

        # Prepare images
        arr = img.astype(np.float64)
        if np.iscomplexobj(img):
            self.orig = np.abs(img).astype(np.float64)
        else:
            self.orig = arr

        mn, mx = float(self.orig.min()), float(self.orig.max())
        if mx > mn:
            disp = ((self.orig - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            disp = np.zeros_like(self.orig, dtype=np.uint8)

        self.orig_h, self.orig_w = self.orig.shape[:2]
        disp8 = np.ascontiguousarray(disp)
        qimg  = QImage(disp8.tobytes(), self.orig_w, self.orig_h, self.orig_w, QImage.Format_Grayscale8)
        self.pix = QPixmap.fromImage(qimg)

        # Window sizing
        screen = QApplication.primaryScreen().availableGeometry()
        screen_h = screen.height()
        screen_w = screen.width()
        initial_size = int(screen_h * 0.8)
        self.resize(initial_size, initial_size)
        self.move((screen_w - initial_size) // 2, (screen_h - initial_size) // 2)
        self.setMinimumSize(400, 400)

        # Widgets
        self.scroll = DragScrollArea(self)
        self.label  = ROIImageLabel()
        self.label.installEventFilter(self)
        self.scroll.setWidget(self.label)

        # WW/WL controls
        self.data_min = mn
        self.data_max = mx
        self.data_range = max(mx - mn, 1e-9)

        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_spin = QSpinBox()
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_spin = QSpinBox()

        self.ww_slider.setRange(1, 10000)
        self.ww_slider.setValue(10000)
        self.ww_spin.setMaximum(999999)
        self.ww_spin.setValue(int(self.data_range))
        self.ww_spin.setSingleStep(max(1, int(self.data_range/100)))

        self.wl_slider.setRange(0, 10000)
        self.wl_slider.setValue(5000)
        self.wl_spin.setMinimum(int(mn)-1000)
        self.wl_spin.setMaximum(int(mx)+1000)
        self.wl_spin.setValue(int((mn+mx)/2))
        self.wl_spin.setSingleStep(max(1, int(self.data_range/100)))

        self.ww_slider.valueChanged.connect(self._update_ww_from_slider)
        self.ww_spin.valueChanged.connect(self._update_ww_from_spin)
        self.wl_slider.valueChanged.connect(self._update_wl_from_slider)
        self.wl_spin.valueChanged.connect(self._update_wl_from_spin)

        # Rubber band & buttons
        self.origin = QPoint()
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.label)
        self.info_label = QLabel("X: -, Y: -, Val: -")
        ok      = QPushButton("Confirm Selection")
        cancel  = QPushButton("Use Full Image")
        save_btn= QPushButton("Save Snapshot")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        save_btn.clicked.connect(self.save_snapshot)

        # Layouts
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll)

        wl_layout = QHBoxLayout()
        for lbl, pair in [("WW:", (self.ww_slider, self.ww_spin)),
                          ("WL:", (self.wl_slider, self.wl_spin))]:
            wl_layout.addWidget(QLabel(lbl))
            wl_layout.addWidget(pair[0]); wl_layout.addWidget(pair[1])
        main_layout.addLayout(wl_layout)

        zb = QHBoxLayout()
        zb.addWidget(QLabel("Zoom:"))
        for btn, fn in [(QPushButton("+"), self.zoom_in),
                        (QPushButton("-"), self.zoom_out),
                        (QPushButton("Reset"), self.zoom_reset)]:
            btn.clicked.connect(fn); zb.addWidget(btn)
        zb.addStretch()
        main_layout.addLayout(zb)

        main_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(ok); btn_layout.addWidget(cancel); btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

        # Auto-fit after widgets exist
        QTimer.singleShot(0, self._auto_fit_image)
        self._render_image()

    # WW/WL handlers
    def _update_ww_from_slider(self, value):
        ww = (value / 10000.0) * self.data_range
        self.ww_spin.blockSignals(True)
        self.ww_spin.setValue(int(max(1, ww)))
        self.ww_spin.blockSignals(False)
        self._render_image()

    def _update_ww_from_spin(self, value):
        slider_val = int((value / self.data_range) * 10000)
        slider_val = max(1, min(10000, slider_val))
        self.ww_slider.blockSignals(True)
        self.ww_slider.setValue(slider_val)
        self.ww_slider.blockSignals(False)
        self._render_image()

    def _update_wl_from_slider(self, value):
        wl = self.data_min + (value / 10000.0) * self.data_range
        self.wl_spin.blockSignals(True)
        self.wl_spin.setValue(int(wl))
        self.wl_spin.blockSignals(False)
        self._render_image()

    def _update_wl_from_spin(self, value):
        slider_val = int(((value - self.data_min) / self.data_range) * 10000)
        slider_val = max(0, min(10000, slider_val))
        self.wl_slider.blockSignals(True)
        self.wl_slider.setValue(slider_val)
        self.wl_slider.blockSignals(False)
        self._render_image()

    def _auto_fit_image(self):
        if not hasattr(self, 'orig') or self.orig is None:
            return
        vp = self.scroll.viewport()
        vp_width = max(vp.width(), 1)
        vp_height = max(vp.height(), 1)
        img_height, img_width = self.orig.shape[:2]
        zoom_x = vp_width / max(img_width, 1)
        zoom_y = vp_height / max(img_height, 1)
        self.zoom = min(zoom_x, zoom_y)
        self._render_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'orig') and self.orig is not None:
            self._auto_fit_image()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        pix_size = self.pixel_size_um
        if pix_size is None or pix_size <= 0:
            return

        vp = self.scroll.viewport()
        W, H = vp.width(), vp.height()
        f = 0.2
        L_raw = (f * W) / max(self.zoom, 1e-9) * pix_size
        if L_raw <= 0:
            return

        min_exp, max_exp = -3, 6
        factors = (1,2,5)
        nice_list = []
        for exponent in range(min_exp, max_exp + 1):
            for factor in factors:
                val = factor * (10 ** exponent)
                if val >= 0.05:
                    nice_list.append(val)
        nice_list.sort()

        L_snap = nice_list[0]
        for candidate in nice_list:
            if candidate <= L_raw:
                L_snap = candidate
            else:
                break

        bar_px = max(1, int(round((L_snap / pix_size) * self.zoom)))
        label_text = f"{L_snap:g} µm"

        x0 = int(W * 0.9) - bar_px
        thickness = max(1, H // 100)
        y0 = H - int(H * 0.04) - thickness
        h_off = self.scroll.horizontalScrollBar().value()
        v_off = self.scroll.verticalScrollBar().value()

        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('white'))
        painter.drawRect(h_off + x0, v_off + y0, bar_px, thickness)
        painter.setPen(QColor('white'))
        painter.drawText(h_off + x0, v_off + y0 - 4, label_text)
        painter.end()

    def _render_image(self):
        ww = max(0.001, float(self.ww_spin.value()))
        wl = float(self.wl_spin.value())
        lo, hi = wl - ww/2.0, wl + ww/2.0
        if hi <= lo:
            hi = lo + 0.001

        norm = np.clip((self.orig - lo) / (hi - lo), 0.0, 1.0)
        img8 = (norm * 255).astype(np.uint8)

        h, w = img8.shape[:2]
        img8c = np.ascontiguousarray(img8)
        qimg = QImage(img8c.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        sw = max(1, int(pix.width() * self.zoom))
        sh = max(1, int(pix.height() * self.zoom))
        scaled = pix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.FastTransformation)

        # display scales used for mapping from widget→image coords
        self.display_scale_x = scaled.width()  / max(self.orig_w, 1)
        self.display_scale_y = scaled.height() / max(self.orig_h, 1)

        # preserve scroll ratios
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        h_ratio = hbar.value() / max(1, hbar.maximum())
        v_ratio = vbar.value() / max(1, vbar.maximum())

        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

        QTimer.singleShot(0, lambda: (
            hbar.setValue(int(h_ratio * hbar.maximum())),
            vbar.setValue(int(v_ratio * vbar.maximum()))
        ))

    def _set_scroll_to_center(self, hval, vval):
        self.scroll.horizontalScrollBar().setValue(int(hval))
        self.scroll.verticalScrollBar().setValue(int(vval))

    def zoom_in(self):
        new_zoom = min(20.0, self.zoom * 1.2)
        self._zoom_at_center(new_zoom)

    def zoom_out(self):
        new_zoom = max(0.2, self.zoom * 0.83)
        self._zoom_at_center(new_zoom)

    def zoom_reset(self):
        self.zoom = 1.0
        self._render_image()
        self.rubberBand.hide()
        QTimer.singleShot(0, lambda: (
            self.scroll.horizontalScrollBar().setValue(0),
            self.scroll.verticalScrollBar().setValue(0)
        ))

    def _zoom_at_center(self, new_zoom):
        vp = self.scroll.viewport()
        center_x = vp.width() / 2
        center_y = vp.height() / 2

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        cx = hbar.value() + center_x
        cy = vbar.value() + center_y

        factor = new_zoom / max(self.zoom, 1e-9)
        self.zoom = new_zoom
        self._render_image()

        new_cx = cx * factor
        new_cy = cy * factor
        QTimer.singleShot(0, lambda: self._set_scroll_to_center(new_cx - center_x, new_cy - center_y))

    def eventFilter(self, src, ev):
        if src is self.label:
            if ev.type() == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self.origin = ev.pos()
                self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                self.rubberBand.show()
                return True
            if ev.type() == QEvent.MouseMove and ev.buttons() & Qt.LeftButton:
                p = ev.pos()
                p.setX(max(0, min(p.x(), self.label.width()-1)))
                p.setY(max(0, min(p.y(), self.label.height()-1)))
                r = QRect(self.origin, p).normalized()
                self.rubberBand.setGeometry(r)
                x = int(r.x() / self.display_scale_x)
                y = int(r.y() / self.display_scale_y)
                w = int(r.width() / self.display_scale_x)
                h = int(r.height() / self.display_scale_y)
                self.info_label.setText(f"Selection: x={x}, y={y}, w={w}, h={h}")
                return True
            if ev.type() == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
                r = QRect(self.origin, ev.pos()).normalized()
                x = int(r.x() / self.display_scale_x)
                y = int(r.y() / self.display_scale_y)
                w = max(1, int(r.width() / self.display_scale_x))
                h = max(1, int(r.height() / self.display_scale_y))
                x = max(0, min(x, self.orig_w - 1))
                y = max(0, min(y, self.orig_h - 1))
                w = max(1, min(w, self.orig_w - x))
                h = max(1, min(h, self.orig_h - y))
                self.selected = QRect(x, y, w, h)
                self.has_selection = (w > 10 and h > 10)
                if self.has_selection:
                    self.info_label.setText(f"Selection confirmed: x={x}, y={y}, w={w}, h={h}")
                else:
                    self.info_label.setText("Selection too small")
                return True
        if ev.type() == QEvent.Wheel:
            steps = ev.angleDelta().y() / 120.0
            base  = 1.2
            old_zoom = self.zoom
            requested = old_zoom * (base ** steps)

            vp = self.scroll.viewport().size()
            orig_h, orig_w = self.pix.height(), self.pix.width()
            min_zoom_x = vp.width() / max(orig_w, 1)
            min_zoom_y = vp.height() / max(orig_h, 1)
            dynamic_min = min(min_zoom_x, min(min_zoom_y, 20.0))
            new_zoom = max(dynamic_min, min(requested, 20.0))

            pos = self.scroll.viewport().mapFromGlobal(ev.globalPos())
            hbar = self.scroll.horizontalScrollBar()
            vbar = self.scroll.verticalScrollBar()

            img_x = (hbar.value() + pos.x()) / max(old_zoom, 1e-9)
            img_y = (vbar.value() + pos.y()) / max(old_zoom, 1e-9)

            self.zoom = new_zoom
            self._render_image()

            new_x = img_x * new_zoom
            new_y = img_y * new_zoom

            QTimer.singleShot(0, lambda: (
                hbar.setValue(int(new_x - pos.x())),
                vbar.setValue(int(new_y - pos.y()))
            ))
            return True
        return super().eventFilter(src, ev)

    def get_roi(self):
        if self.has_selection:
            r = self.selected
            return [r.x(), r.y(), r.width(), r.height()]
        return [0, 0, self.orig_w, self.orig_h]

    def save_snapshot(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI Snapshot",
            "",
            "PNG Files (*.png);;All Files (*)",
            options=options
        )
        if not filename:
            return
        if not filename.lower().endswith('.png'):
            filename += '.png'
        pixmap = self.scroll.viewport().grab()
        pixmap.save(filename, "PNG")

class ROIImageLabel(QLabel):
    """Scale bar inside the ROI selector, snapping to 0.05→0.1→0.5→1→… μm."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        win = self.window()
        if not isinstance(win, ROISelector):
            return

        pix_size = win.pixel_size_um
        if pix_size is None or pix_size <= 0:
            return

        vp = win.scroll.viewport()
        W, H = vp.width(), vp.height()
        f = 0.2
        L_raw = (f * W) / max(win.zoom, 1e-9) * pix_size
        if L_raw <= 0:
            return

        min_exp, max_exp = -3, 6
        factors = (1,2,5)
        nice_list = []
        for exponent in range(min_exp, max_exp + 1):
            for factor in factors:
                val = factor * (10 ** exponent)
                if val >= 0.05:
                    nice_list.append(val)
        nice_list.sort()

        L_snap = nice_list[0]
        for candidate in nice_list:
            if candidate <= L_raw:
                L_snap = candidate
            else:
                break

        bar_px = max(1, int(round((L_snap / pix_size) * win.zoom)))
        label_text = f"{L_snap:g} µm"

        x0 = int(W * 0.9) - bar_px
        thickness = max(1, H // 100)
        y0 = H - int(H * 0.04) - thickness

        h_off = win.scroll.horizontalScrollBar().value()
        v_off = win.scroll.verticalScrollBar().value()

        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('white'))
        painter.drawRect(h_off + x0, v_off + y0, bar_px, thickness)
        painter.setPen(QColor('white'))
        painter.drawText(h_off + x0, v_off + y0 - 4, label_text)
        painter.end()

    def mouseMoveEvent(self, ev):
        win = self.window()
        if not isinstance(win, ROISelector):
            return super().mouseMoveEvent(ev)

        ix = int(ev.x() / max(getattr(win, 'display_scale_x', 1.0), 1e-9))
        iy = int(ev.y() / max(getattr(win, 'display_scale_y', 1.0), 1e-9))
        h, w = win.orig.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            val = win.orig[iy, ix]
            if isinstance(val, (np.complexfloating, complex)):
                text = f"X: {ix}, Y: {iy}, |Val|: {abs(val):.3f}"
            else:
                text = f"X: {ix}, Y: {iy}, Val: {float(val):.3f}"
        else:
            text = f"X: {ix}, Y: {iy} (out of bounds)"
        win.info_label.setText(text)
        return super().mouseMoveEvent(ev)

def display_image(img, title="Image", pixel_size=None, roi=False):
    """
    Unified image display / ROI selector.

    Parameters:
    - img: numpy array image
    - title: window title
    - pixel_size: pixel size in microns for scale bar
    - roi: if True, opens ROI selector dialog and returns [x,y,w,h]

    Returns:
    - If roi=False: ImageWindow instance
    - If roi=True: [x, y, w, h] ROI or full image bounds
    """
    app = get_application()
    if roi:
        dlg = ROISelector(img, pixel_size_um=pixel_size)
        dlg.setWindowTitle(title)
        if dlg.exec_() == QDialog.Accepted and dlg.has_selection:
            roi_coords = dlg.get_roi()
            # enforce even dimensions
            roi_coords[2] -= roi_coords[2] % 2
            roi_coords[3] -= roi_coords[3] % 2
            if roi_coords[2] > 0 and roi_coords[3] > 0:
                return roi_coords
        return [0, 0, img.shape[1], img.shape[0]]
    else:
        win = ImageWindow(img, title)
        win.show()
        _windows.append(win)  # keep a ref
        app.processEvents()
        win.label.update()
        win.scroll.viewport().update()
        return win

# Convenience
def process_events():
    get_application().processEvents()

def loop():
    return get_application().exec_()

if __name__ == "__main__":
    sys.exit(loop())
