#!/usr/bin/env python3
"""
Information Theory — Interactive Learning App
=============================================
A comprehensive interactive learning application covering fundamental and
advanced topics in Information Theory, built with PySide6, Matplotlib, and NumPy.

Topics covered:
  1. Dashboard
  2. Shannon Entropy
  3. Joint & Conditional Entropy
  4. Mutual Information
  5. Huffman Coding
  6. Channel Capacity (BSC)
  7. KL Divergence & Cross Entropy
  8. Source Coding Theorem
  9. Practice Quiz
  10. Noisy Channel Coding Theorem
  11. Fano's Inequality
  12. Hamming(7,4) Code
  13. Arithmetic Coding
  14. Markov Chain Entropy Rate
  15. Formula Reference

Installation:
  pip install PySide6 numpy matplotlib

Usage:
  python info_theory_app.py
"""

import sys
import math
import random
import heapq

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QLineEdit, QTextEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QListWidget, QStackedWidget,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MplCircle


# ═══════════════════════════════════════════════════════════════════════════════
#  DARK THEME STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

STYLESHEET = """
QMainWindow { background-color: #0f0f1a; }
QWidget { background-color: #0f0f1a; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; font-size: 13px; }

QListWidget {
    background-color: transparent; border: none; outline: none; padding: 6px 4px;
}
QListWidget::item { padding: 10px 14px; border-radius: 8px; margin: 1px 4px; color: #7f849c; font-size: 13px; }
QListWidget::item:selected { background-color: #2d2d50; color: #89b4fa; font-weight: bold; }
QListWidget::item:hover  { background-color: #1e1e35; color: #a6adc8; }

QLabel { background: transparent; }

QGroupBox {
    background-color: #151525; border: 1px solid #2a2a40; border-radius: 10px;
    margin-top: 16px; padding: 16px; padding-top: 28px;
    font-weight: bold; color: #89b4fa;
}
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; }

QPushButton {
    background-color: #89b4fa; color: #0f0f1a; border: none; border-radius: 8px;
    padding: 9px 22px; font-weight: bold; font-size: 13px;
}
QPushButton:hover { background-color: #a6c8ff; }
QPushButton:pressed { background-color: #6a9be6; }
QPushButton#secondary {
    background-color: #2a2a40; color: #89b4fa; border: 1px solid #3a3a55;
}
QPushButton#secondary:hover { background-color: #3a3a55; }

QSlider::groove:horizontal { height: 6px; background: #2a2a40; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #89b4fa; width: 18px; height: 18px; margin: -6px 0; border-radius: 9px;
}
QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }

QLineEdit {
    background-color: #151525; border: 1px solid #2a2a40; border-radius: 6px;
    padding: 8px 12px; color: #cdd6f4; font-family: 'Consolas', 'Courier New', monospace;
}
QLineEdit:focus { border-color: #89b4fa; }

QTextEdit {
    background-color: #12121f; border: 1px solid #2a2a40; border-radius: 6px;
    padding: 8px; color: #cdd6f4; font-family: 'Consolas', 'Courier New', monospace;
}

QTableWidget {
    background-color: #151525; border: 1px solid #2a2a40; border-radius: 6px;
    gridline-color: #2a2a40; color: #cdd6f4;
}
QTableWidget::item { padding: 4px; }
QHeaderView::section {
    background-color: #1e1e35; color: #89b4fa; border: 1px solid #2a2a40;
    padding: 5px; font-weight: bold;
}

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0f0f1a; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #3a3a55; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #0f0f1a; height: 8px; border-radius: 4px; }
QScrollBar::handle:horizontal { background: #3a3a55; border-radius: 4px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS
# ═══════════════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    """Matplotlib canvas with dark-theme defaults."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0f0f1a')
        self.axes = self.fig.add_subplot(111)
        self._style(self.axes)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @staticmethod
    def _style(ax):
        ax.set_facecolor('#151525')
        ax.tick_params(colors='#7f849c', labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#2a2a40')
        ax.xaxis.label.set_color('#7f849c')
        ax.yaxis.label.set_color('#7f849c')
        ax.title.set_color('#89b4fa')


# ═══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def shannon_entropy(p):
    """H(X) for a probability distribution (bits)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def binary_entropy(p):
    """Binary entropy function H_b(p) (bits)."""
    p = float(np.clip(p, 1e-15, 1 - 1e-15))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def kl_divergence(p, q):
    """D_KL(P || Q) in bits."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

def cross_entropy(p, q):
    """H(P, Q) in bits."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = (p > 0) & (q > 0)
    return float(-np.sum(p[mask] * np.log2(q[mask])))


# ═══════════════════════════════════════════════════════════════════════════════
#  HUFFMAN HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class HuffmanNode:
    def __init__(self, symbol=None, prob=0.0, left=None, right=None):
        self.symbol = symbol
        self.prob = prob
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(symbols_probs):
    heap = [HuffmanNode(s, p) for s, p in symbols_probs]
    heapq.heapify(heap)
    steps = []
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(prob=left.prob + right.prob, left=left, right=right)
        ll = left.symbol if left.symbol else f"({left.prob:.3f})"
        rl = right.symbol if right.symbol else f"({right.prob:.3f})"
        steps.append((ll, rl, merged.prob))
        heapq.heappush(heap, merged)
    return (heap[0] if heap else None), steps


def get_huffman_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if node is None:
        return codes
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else "0"
        return codes
    get_huffman_codes(node.left, prefix + "0", codes)
    get_huffman_codes(node.right, prefix + "1", codes)
    return codes


def draw_huffman_tree(node, ax, x=0.5, y=1.0, dx=0.25, dy=0.14):
    if node is None:
        return
    if node.symbol is not None:
        ax.text(x, y, f"{node.symbol}\n{node.prob:.3f}", ha='center', va='center',
                fontsize=9, fontweight='bold', color='#a6e3a1',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a3a2a',
                          edgecolor='#a6e3a1', alpha=0.9))
    else:
        ax.text(x, y, f"{node.prob:.3f}", ha='center', va='center',
                fontsize=8, color='#fab387',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#2a2a10',
                          edgecolor='#fab387', alpha=0.7))
    if node.left:
        nx, ny = x - dx, y - dy
        ax.annotate('', xy=(nx, ny + 0.03), xytext=(x, y - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#585b70', lw=1.5))
        ax.text((x + nx) / 2 - 0.02, (y + ny) / 2, '0', fontsize=8, color='#7f849c')
        draw_huffman_tree(node.left, ax, nx, ny, dx * 0.55, dy)
    if node.right:
        nx, ny = x + dx, y - dy
        ax.annotate('', xy=(nx, ny + 0.03), xytext=(x, y - 0.02),
                    arrowprops=dict(arrowstyle='->', color='#585b70', lw=1.5))
        ax.text((x + nx) / 2 + 0.02, (y + ny) / 2, '1', fontsize=8, color='#7f849c')
        draw_huffman_tree(node.right, ax, nx, ny, dx * 0.55, dy)


# ═══════════════════════════════════════════════════════════════════════════════
#  REUSABLE UI WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

def title_label(text):
    l = QLabel(text)
    l.setStyleSheet("font-size: 28px; font-weight: bold; color: #89b4fa;")
    return l

def formula_label(text):
    l = QLabel(text)
    l.setStyleSheet("font-size: 15px; color: #fab387; font-family: 'Consolas','Courier New',monospace; padding: 8px;")
    return l

def info_label(text):
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet("color: #a6adc8; font-size: 13px; line-height: 1.5;")
    return l

def result_label(text=""):
    l = QLabel(text)
    l.setStyleSheet("font-family:'Consolas','Courier New',monospace; font-size: 16px; color: #a6e3a1; font-weight: bold;")
    return l

def detail_textedit():
    te = QTextEdit()
    te.setReadOnly(True)
    te.setStyleSheet(
        "background-color: #12121f; border: 1px solid #2a2a40; border-radius: 8px; "
        "padding: 8px; color: #cdd6f4; font-family: 'Consolas','Courier New',monospace; font-size: 11px;"
    )
    return te


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(20)
        lay.setContentsMargins(40, 30, 40, 30)

        lay.addWidget(title_label("Information Theory"))
        sub = QLabel("An Interactive Learning Experience")
        sub.setStyleSheet("font-size: 16px; color: #7f849c;")
        lay.addWidget(sub)
        lay.addWidget(info_label(
            "Information theory is a mathematical framework for quantifying, storing, "
            "and communicating information. Founded by Claude Shannon in 1948, it underpins "
            "modern digital communications, data compression, and cryptography.\n\n"
            "Navigate the topics on the left sidebar to explore each concept interactively."
        ))
        lay.addSpacing(15)

        topics = [
            ("📐", "Shannon Entropy", "The fundamental measure of information & uncertainty"),
            ("🔗", "Joint & Conditional Entropy", "Information in correlated random variables"),
            ("🔄", "Mutual Information", "How much one variable reveals about another"),
            ("🌲", "Huffman Coding", "Optimal prefix-free coding for data compression"),
            ("📡", "Channel Capacity", "Maximum reliable communication rate over noisy channels"),
            ("📏", "KL Divergence", "Measuring difference between probability distributions"),
            ("💡", "Source Coding Theorem", "The theoretical limit of data compression"),
            ("🧪", "Practice Quiz", "Test your understanding with interactive questions"),
            ("📡", "Noisy Channel Theorem", "Reliable communication in the presence of noise"),
            ("📐", "Fano's Inequality", "Connecting estimation error to conditional entropy"),
            ("🛡️", "Hamming Code", "Error detection and correction visualized"),
            ("🔢", "Arithmetic Coding", "Encoding messages as fractional intervals"),
            ("⛓️", "Markov Entropy Rate", "Entropy rate for correlated symbol sequences"),
            ("📖", "Formula Reference", "Complete cheat sheet of all key formulas"),
        ]

        grid = QGridLayout()
        grid.setSpacing(14)
        for i, (icon, name, desc) in enumerate(topics):
            card = QFrame()
            card.setStyleSheet("""
                QFrame { background-color: #151525; border: 1px solid #2a2a40;
                         border-radius: 12px; padding: 16px; }
                QFrame:hover { border-color: #89b4fa; background-color: #1a1a30; }
            """)
            cl = QVBoxLayout(card)
            cl.setSpacing(4)
            il = QLabel(icon)
            il.setStyleSheet("font-size: 26px; background: transparent;")
            cl.addWidget(il)
            nl = QLabel(name)
            nl.setStyleSheet("font-size: 13px; font-weight: bold; color: #89b4fa; background: transparent;")
            cl.addWidget(nl)
            dl = QLabel(desc)
            dl.setWordWrap(True)
            dl.setStyleSheet("font-size: 11px; color: #7f849c; background: transparent;")
            cl.addWidget(dl)
            grid.addWidget(card, i // 3, i % 3)

        lay.addLayout(grid)
        lay.addStretch()

        quote = QLabel(
            '"The fundamental problem of communication is that of reproducing at one point\n'
            'either exactly or approximately a message selected at another point."\n'
            '— Claude Shannon, 1948'
        )
        quote.setAlignment(Qt.AlignCenter)
        quote.setStyleSheet("font-style: italic; color: #585b70; font-size: 13px; padding: 20px;")
        lay.addWidget(quote)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SHANNON ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Shannon Entropy"))
        lay.addWidget(formula_label("H(X) = −Σ  p(x) · log₂ p(x)"))
        lay.addWidget(info_label(
            "Shannon entropy measures the average uncertainty (information content) of a random variable. "
            "A uniform distribution has maximum entropy; a deterministic variable has zero entropy. "
            "Drag the slider to explore the binary entropy function, or enter your own distribution."
        ))

        content = QHBoxLayout()

        # ── Left ──
        left = QWidget()
        ll = QVBoxLayout(left)

        bg = QGroupBox("Binary Entropy Function  H(p)")
        bl = QVBoxLayout(bg)
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 999)
        self.p_slider.setValue(500)
        self.p_slider.valueChanged.connect(self._update_binary)
        self.p_label = QLabel("p = 0.500")
        self.p_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px;")
        bl.addWidget(self.p_label)
        bl.addWidget(self.p_slider)
        self.h_bin = result_label("H(X) = 1.0000 bits")
        bl.addWidget(self.h_bin)
        self.bin_canvas = MplCanvas(self, width=5, height=3.5)
        bl.addWidget(self.bin_canvas)
        ll.addWidget(bg)

        cg = QGroupBox("Custom Distribution")
        cl = QVBoxLayout(cg)
        hint = QLabel("Enter probabilities (comma-separated, sum = 1):")
        hint.setStyleSheet("color: #7f849c; font-size: 12px;")
        cl.addWidget(hint)
        self.prob_input = QLineEdit("0.4, 0.3, 0.2, 0.1")
        cl.addWidget(self.prob_input)
        btn = QPushButton("Calculate Entropy")
        btn.clicked.connect(self._calc_custom)
        cl.addWidget(btn)
        self.custom_res = result_label()
        cl.addWidget(self.custom_res)
        self.custom_canvas = MplCanvas(self, width=5, height=3)
        cl.addWidget(self.custom_canvas)
        ll.addWidget(cg)
        content.addWidget(left, 3)

        # ── Right: step-by-step ──
        right = QWidget()
        rl = QVBoxLayout(right)
        sg = QGroupBox("Step-by-Step Calculation")
        sl = QVBoxLayout(sg)
        self.steps_text = detail_textedit()
        self.steps_text.setMinimumWidth(340)
        sl.addWidget(self.steps_text)
        rl.addWidget(sg)
        rl.addStretch()
        content.addWidget(right, 2)

        lay.addLayout(content)
        lay.addStretch()
        self._update_binary()
        self._calc_custom()

    def _update_binary(self):
        p = self.p_slider.value() / 1000.0
        self.p_label.setText(f"p = {p:.3f}")
        h = binary_entropy(p)
        self.h_bin.setText(f"H(X) = {h:.4f} bits")
        ax = self.bin_canvas.axes
        ax.clear()
        x = np.linspace(0.001, 0.999, 500)
        y = np.array([binary_entropy(xi) for xi in x])
        ax.plot(x, y, color='#89b4fa', linewidth=2.5, label='H(p)')
        ax.axvline(x=p, color='#fab387', linestyle='--', alpha=0.7, label=f'p = {p:.3f}')
        ax.plot(p, h, 'o', color='#fab387', markersize=10, zorder=5)
        ax.set_xlabel('p'); ax.set_ylabel('H(p) [bits]')
        ax.set_title('Binary Entropy Function')
        ax.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.15, color='#3a3a55')
        self.bin_canvas.fig.tight_layout()
        self.bin_canvas.draw()

    def _calc_custom(self):
        try:
            probs = [float(x.strip()) for x in self.prob_input.text().split(',') if x.strip()]
            if abs(sum(probs) - 1.0) > 0.01:
                self.custom_res.setText("⚠  Probabilities must sum to 1.0"); return
            if any(p < 0 for p in probs):
                self.custom_res.setText("⚠  Probabilities must be ≥ 0"); return
            h = shannon_entropy(probs)
            hmax = math.log2(len(probs))

            steps = [f"{'═'*40}\n  Step-by-Step Entropy Calculation\n{'═'*40}\n\n"]
            steps.append(f"Distribution: [{', '.join(f'{p:.4f}' for p in probs)}]\n")
            steps.append(f"Number of symbols: {len(probs)}\n")
            steps.append(f"Maximum entropy: log₂({len(probs)}) = {hmax:.4f} bits\n\n")
            steps.append(f"{'─'*40}\n  Individual terms\n{'─'*40}\n")
            for i, p in enumerate(probs):
                if p > 0:
                    t = -p * math.log2(p)
                    steps.append(f"  p(x{i}) = {p:.4f}  →  −{p:.4f}·log₂({p:.4f}) = {t:.4f}\n")
                else:
                    steps.append(f"  p(x{i}) = 0.0000  →  0  (by convention)\n")
            steps.append(f"\n{'─'*40}\n  Result\n{'─'*40}\n")
            steps.append(f"  H(X)       = {h:.4f} bits\n")
            steps.append(f"  H_max      = {hmax:.4f} bits\n")
            steps.append(f"  Efficiency = {h / hmax * 100:.1f}%\n")
            self.steps_text.setText(''.join(steps))
            self.custom_res.setText(
                f"H(X) = {h:.4f} bits   |   Max = {hmax:.4f}   |   η = {h / hmax * 100:.1f}%"
            )

            ax = self.custom_canvas.axes; ax.clear()
            syms = [f'x{i}' for i in range(len(probs))]
            colors = ['#89b4fa', '#fab387', '#a6e3a1', '#f38ba8', '#cba6f7',
                      '#94e2d5', '#f9e2af', '#74c7ec', '#eba0ac', '#89dceb']
            ax.bar(syms, probs, color=colors[:len(probs)], alpha=0.85, edgecolor='#ffffff10')
            ax.text(0.98, 0.95, f'H(X) = {h:.4f} bits', transform=ax.transAxes,
                    ha='right', va='top', fontsize=11, color='#a6e3a1',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a3a2a',
                              edgecolor='#a6e3a1', alpha=0.8))
            ax.set_xlabel('Symbol'); ax.set_ylabel('Probability')
            ax.set_title('Probability Distribution')
            ax.set_ylim(0, max(probs) * 1.3 if probs else 1)
            ax.grid(True, alpha=0.15, color='#3a3a55', axis='y')
            self.custom_canvas.fig.tight_layout(); self.custom_canvas.draw()
        except ValueError:
            self.custom_res.setText("⚠  Invalid input — enter comma-separated numbers")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — JOINT & CONDITIONAL ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

class JointEntropyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Joint & Conditional Entropy"))
        lay.addWidget(formula_label(
            "H(X,Y) = −ΣΣ p(x,y)·log₂ p(x,y)\n"
            "H(X|Y) = H(X,Y) − H(Y)       H(Y|X) = H(X,Y) − H(X)"
        ))
        lay.addWidget(info_label(
            "Joint entropy H(X,Y) measures total uncertainty in two random variables. "
            "Conditional entropy H(X|Y) measures remaining uncertainty in X after observing Y. "
            "Edit the joint probability table to see how they change."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        tg = QGroupBox("Joint Probability Table  P(X, Y)")
        tl = QVBoxLayout(tg)
        self.table = QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(["Y=0", "Y=1", "P(X)"])
        self.table.setVerticalHeaderLabels(["X=0", "X=1", "P(Y)"])
        default = [["0.25", "0.25", ""], ["0.25", "0.25", ""], ["", "", ""]]
        for i in range(3):
            for j in range(3):
                it = QTableWidgetItem(default[i][j])
                it.setTextAlignment(Qt.AlignCenter)
                if i == 2 or j == 2:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setBackground(QColor("#1a1a30"))
                self.table.setItem(i, j, it)
        self.table.setMaximumHeight(150)
        self.table.cellChanged.connect(self._update)
        tl.addWidget(self.table)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._update); tl.addWidget(btn)
        ll.addWidget(tg)

        rg = QGroupBox("Results")
        rl = QVBoxLayout(rg)
        self.lbl_hx = result_label("H(X)  = --")
        self.lbl_hy = QLabel("H(Y)  = --")
        self.lbl_hy.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#fab387; font-weight:bold;")
        self.lbl_hxy = QLabel("H(X,Y)= --")
        self.lbl_hxy.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#cba6f7; font-weight:bold;")
        self.lbl_hxgy = QLabel("H(X|Y)= --")
        self.lbl_hxgy.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#a6e3a1; font-weight:bold;")
        self.lbl_hygx = QLabel("H(Y|X)= --")
        self.lbl_hygx.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#f9e2af; font-weight:bold;")
        self.lbl_mi = QLabel("I(X;Y)= --")
        self.lbl_mi.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#f38ba8; font-weight:bold;")
        for w in (self.lbl_hx, self.lbl_hy, self.lbl_hxy, self.lbl_hxgy, self.lbl_hygx, self.lbl_mi):
            rl.addWidget(w)
        ll.addWidget(rg)
        content.addWidget(left, 1)

        right = QWidget(); rl2 = QVBoxLayout(right)
        self.venn_canvas = MplCanvas(self, width=5, height=5)
        rl2.addWidget(self.venn_canvas)
        content.addWidget(right, 1)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        try:
            vals = []
            for i in range(2):
                row = []
                for j in range(2):
                    row.append(float(self.table.item(i, j).text()))
                vals.append(row)
            joint = np.array(vals)
            px = joint.sum(axis=1); py = joint.sum(axis=0)
            for i in range(2):
                self.table.item(i, 2).setText(f"{px[i]:.4f}")
                self.table.item(2, i).setText(f"{py[i]:.4f}")
            self.table.item(2, 2).setText(f"{joint.sum():.4f}")
            if abs(joint.sum() - 1.0) > 0.01:
                self.lbl_hxy.setText("⚠  Must sum to 1"); return
            hx = shannon_entropy(px); hy = shannon_entropy(py)
            hxy = shannon_entropy(joint.flatten())
            hxgy = hxy - hy; hygx = hxy - hx; mi = hx + hy - hxy
            self.lbl_hx.setText(f"H(X)   = {hx:.4f} bits")
            self.lbl_hy.setText(f"H(Y)   = {hy:.4f} bits")
            self.lbl_hxy.setText(f"H(X,Y) = {hxy:.4f} bits")
            self.lbl_hxgy.setText(f"H(X|Y) = {hxgy:.4f} bits")
            self.lbl_hygx.setText(f"H(Y|X) = {hygx:.4f} bits")
            self.lbl_mi.setText(f"I(X;Y) = {mi:.4f} bits")

            ax = self.venn_canvas.axes; ax.clear()
            d = 1.4; r = 1.0
            c1 = MplCircle((-d/2, 0), r, facecolor='#89b4fa20', edgecolor='#89b4fa', linewidth=2)
            c2 = MplCircle((d/2, 0), r, facecolor='#fab38720', edgecolor='#fab387', linewidth=2)
            ax.add_patch(c1); ax.add_patch(c2)
            ax.text(-d/2 - 0.35, 0, f'H(X)\n{hx:.3f}', ha='center', va='center',
                    fontsize=11, color='#89b4fa', fontweight='bold')
            ax.text(d/2 + 0.35, 0, f'H(Y)\n{hy:.3f}', ha='center', va='center',
                    fontsize=11, color='#fab387', fontweight='bold')
            if mi > 0.001:
                ax.text(0, 0.15, f'I(X;Y)\n{mi:.3f}', ha='center', va='center',
                        fontsize=10, color='#a6e3a1', fontweight='bold')
            if hxgy > 0.001:
                ax.text(-d/2, -0.7, f'H(X|Y)\n{hxgy:.3f}', ha='center', fontsize=9, color='#7f849c')
            if hygx > 0.001:
                ax.text(d/2, -0.7, f'H(Y|X)\n{hygx:.3f}', ha='center', fontsize=9, color='#7f849c')
            ax.text(0, 1.5, f'H(X,Y) = {hxy:.3f} bits', ha='center', fontsize=13,
                    color='#cdd6f4', fontweight='bold')
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 2.2); ax.set_aspect('equal'); ax.axis('off')
            ax.set_title('Entropy Venn Diagram', fontsize=13, pad=10)
            self.venn_canvas.fig.tight_layout(); self.venn_canvas.draw()
        except (ValueError, AttributeError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

class MutualInfoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Mutual Information"))
        lay.addWidget(formula_label(
            "I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X) = H(X)+H(Y)−H(X,Y)"
        ))
        lay.addWidget(info_label(
            "Mutual information measures the amount of information one random variable "
            "reveals about another. It is symmetric and non-negative: I(X;Y) ≥ 0, "
            "with equality iff X and Y are independent."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)
        cg = QGroupBox("Bivariate Gaussian (Discretized)")
        cl = QVBoxLayout(cg)
        self.rho_slider = QSlider(Qt.Horizontal)
        self.rho_slider.setRange(0, 99); self.rho_slider.setValue(70)
        self.rho_slider.valueChanged.connect(self._update)
        self.rho_label = QLabel("ρ = 0.700")
        self.rho_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px;")
        cl.addWidget(self.rho_label); cl.addWidget(self.rho_slider)
        self.mi_label = result_label("I(X;Y) = --")
        cl.addWidget(self.mi_label)
        self.mi_detail = detail_textedit()
        self.mi_detail.setMaximumHeight(200)
        cl.addWidget(self.mi_detail)
        ll.addWidget(cg)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.mi_canvas = MplCanvas(self, width=6, height=5)
        rl.addWidget(self.mi_canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        rho = self.rho_slider.value() / 100.0
        self.rho_label.setText(f"ρ = {rho:.3f}")
        n = 60; x = np.linspace(-3, 3, n); dx = x[1] - x[0]
        X, Y = np.meshgrid(x, x)
        det = max(1 - rho**2, 1e-8)
        Z = np.exp(-(X**2 - 2*rho*X*Y + Y**2) / (2*det)) / (2*np.pi*np.sqrt(det))
        joint = Z * dx * dx; joint /= joint.sum()
        px = joint.sum(axis=1); py = joint.sum(axis=0)
        hx = shannon_entropy(px); hy = shannon_entropy(py)
        hxy = shannon_entropy(joint.flatten()); mi = hx + hy - hxy
        self.mi_label.setText(f"I(X;Y) = {mi:.4f} bits")

        txt = [f"{'═'*38}\n  Mutual Information Breakdown\n{'═'*38}\n\n"]
        txt.append(f"Correlation  ρ = {rho:.3f}\n\n")
        txt.append(f"H(X)   = {hx:.4f} bits\n")
        txt.append(f"H(Y)   = {hy:.4f} bits\n")
        txt.append(f"H(X,Y) = {hxy:.4f} bits\n\n")
        txt.append(f"H(X|Y) = {hxy-hy:.4f} bits\n")
        txt.append(f"H(Y|X) = {hxy-hx:.4f} bits\n\n")
        txt.append(f"I(X;Y) = H(X)+H(Y)−H(X,Y)\n")
        txt.append(f"       = {hx:.4f}+{hy:.4f}−{hxy:.4f}\n")
        txt.append(f"       = {mi:.4f} bits\n")
        if abs(rho) < 0.01:
            txt.append("\n✓ ρ ≈ 0 → X,Y nearly independent → I ≈ 0")
        elif abs(rho) > 0.9:
            txt.append("\n✓ ρ ≈ 1 → X,Y highly correlated → I large")
        else:
            txt.append("\n✓ Moderate correlation → some shared info")
        self.mi_detail.setText(''.join(txt))

        ax = self.mi_canvas.axes; ax.clear()
        ax.imshow(joint, extent=[-3, 3, -3, 3], origin='lower',
                  cmap='magma', aspect='auto', interpolation='bilinear')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title(f'Joint Distribution  (ρ = {rho:.2f})   I(X;Y) = {mi:.3f} bits')
        self.mi_canvas.fig.tight_layout(); self.mi_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — HUFFMAN CODING
# ═══════════════════════════════════════════════════════════════════════════════

class HuffmanPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Huffman Coding"))
        lay.addWidget(formula_label("L̄ = Σ p(xᵢ)·l(xᵢ)  ≥  H(X)   (Source Coding Theorem)"))
        lay.addWidget(info_label(
            "Huffman coding is an optimal prefix-free coding algorithm. It assigns shorter "
            "codes to more frequent symbols. The average code length approaches entropy."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        ig = QGroupBox("Symbol Probabilities")
        il = QVBoxLayout(ig)
        hint = QLabel("Format:  symbol:probability  (one per line)")
        hint.setStyleSheet("color:#7f849c; font-size:11px;")
        il.addWidget(hint)
        self.sym_input = QTextEdit()
        self.sym_input.setMaximumHeight(160)
        self.sym_input.setPlainText("A:0.35\nB:0.25\nC:0.20\nD:0.12\nE:0.08")
        il.addWidget(self.sym_input)

        pl = QHBoxLayout()
        for name, vals in [("Uniform", "A:0.25\nB:0.25\nC:0.25\nD:0.25"),
                           ("Skewed", "A:0.50\nB:0.25\nC:0.15\nD:0.10"),
                           ("5-symbol", "A:0.35\nB:0.25\nC:0.20\nD:0.12\nE:0.08")]:
            b = QPushButton(name); b.setObjectName("secondary")
            b.clicked.connect(lambda _, v=vals: (self.sym_input.setPlainText(v), self._build()))
            pl.addWidget(b)
        il.addLayout(pl)

        btn = QPushButton("Build Huffman Tree"); btn.clicked.connect(self._build); il.addWidget(btn)
        ll.addWidget(ig)

        rg = QGroupBox("Coding Results")
        rl = QVBoxLayout(rg)
        self.huff_result = detail_textedit()
        rl.addWidget(self.huff_result)
        ll.addWidget(rg)
        content.addWidget(left, 2)

        right = QWidget(); rl2 = QVBoxLayout(right)
        self.tree_canvas = MplCanvas(self, width=6, height=6)
        rl2.addWidget(self.tree_canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._build()

    def _build(self):
        try:
            lines = self.sym_input.toPlainText().strip().split('\n')
            sp = []
            for line in lines:
                if ':' in line:
                    s, p = line.split(':')
                    sp.append((s.strip(), float(p.strip())))
            if not sp:
                self.huff_result.setText("⚠  No valid symbols"); return
            if abs(sum(p for _, p in sp) - 1.0) > 0.01:
                self.huff_result.setText("⚠  Probabilities must sum to 1.0"); return
            root, steps = build_huffman_tree(sp)
            codes = get_huffman_codes(root)
            h = shannon_entropy([p for _, p in sp])
            avg = sum(p * len(codes[s]) for s, p in sp if s in codes)
            eff = h / avg * 100 if avg > 0 else 0

            r = [f"{'═'*38}\n  Huffman Coding Results\n{'═'*38}\n\n"]
            r.append(f"Shannon Entropy H(X) = {h:.4f} bits\n\n")
            r.append(f"{'─'*38}\n  Codes\n{'─'*38}\n")
            for s, p in sorted(sp, key=lambda x: -x[1]):
                c = codes.get(s, "?")
                r.append(f"  {s}  p={p:.3f}  code={c}  len={len(c)}\n")
            r.append(f"\n{'─'*38}\n  Summary\n{'─'*38}\n")
            r.append(f"  Avg code length L̄  = {avg:.4f} bits\n")
            r.append(f"  Entropy       H(X) = {h:.4f} bits\n")
            r.append(f"  Redundancy  L̄−H    = {avg-h:.4f} bits\n")
            r.append(f"  Efficiency         = {eff:.1f}%\n")
            r.append(f"\n{'─'*38}\n  Build Steps\n{'─'*38}\n")
            for i, (l, ri, mp) in enumerate(steps):
                r.append(f"  {i+1}. Merge [{l}] + [{ri}] → prob={mp:.3f}\n")
            self.huff_result.setText(''.join(r))

            ax = self.tree_canvas.axes; ax.clear(); ax.axis('off')
            draw_huffman_tree(root, ax, x=0.5, y=0.95, dx=0.22, dy=0.14)
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
            ax.set_title('Huffman Tree', fontsize=13, pad=10)
            self.tree_canvas.fig.tight_layout(); self.tree_canvas.draw()
        except Exception as e:
            self.huff_result.setText(f"⚠  Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — CHANNEL CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════

class ChannelCapacityPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Channel Capacity"))
        lay.addWidget(formula_label(
            "C = max I(X;Y)   [bits/channel use]\n"
            "BSC:  C = 1 − H(p)   where p = crossover probability"
        ))
        lay.addWidget(info_label(
            "Channel capacity is the maximum rate at which information can be reliably "
            "transmitted. For the Binary Symmetric Channel (BSC), capacity decreases "
            "with increasing crossover probability p."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        bg = QGroupBox("Binary Symmetric Channel")
        bl = QVBoxLayout(bg)
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(0, 500); self.p_slider.setValue(50)
        self.p_slider.valueChanged.connect(self._update)
        self.p_label = QLabel("p = 0.050")
        self.p_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px;")
        bl.addWidget(self.p_label); bl.addWidget(self.p_slider)
        self.cap_label = result_label("C = 0.7141 bits/use")
        bl.addWidget(self.cap_label)

        self.bsc_canvas = MplCanvas(self, width=4.5, height=3)
        bl.addWidget(self.bsc_canvas)

        self.ch_info = detail_textedit()
        self.ch_info.setMaximumHeight(180)
        bl.addWidget(self.ch_info)
        ll.addWidget(bg)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.cap_canvas = MplCanvas(self, width=6, height=5)
        rl.addWidget(self.cap_canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        p = self.p_slider.value() / 1000.0
        self.p_label.setText(f"Crossover probability p = {p:.3f}")
        hp = binary_entropy(p); cap = 1 - hp
        self.cap_label.setText(f"C = {cap:.4f} bits/use")

        # BSC diagram
        ax = self.bsc_canvas.axes; ax.clear(); ax.axis('off')
        ax.set_xlim(-1, 11); ax.set_ylim(-1, 7)
        for lx, ly, c, lbl in [(1,5,'#89b4fa','0'), (1,2,'#89b4fa','1'),
                                 (9,5,'#a6e3a1','0'), (9,2,'#a6e3a1','1')]:
            ax.text(lx, ly, lbl, fontsize=16, ha='center', va='center', color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f0f1a', edgecolor=c))
        ax.annotate('', xy=(8.2,5), xytext=(1.8,5), arrowprops=dict(arrowstyle='->', color='#89b4fa', lw=2.5))
        ax.text(5, 5.3, f'1−p = {1-p:.3f}', ha='center', fontsize=10, color='#89b4fa')
        ax.annotate('', xy=(8.2,2), xytext=(1.8,2), arrowprops=dict(arrowstyle='->', color='#89b4fa', lw=2.5))
        ax.text(5, 1.7, f'1−p = {1-p:.3f}', ha='center', fontsize=10, color='#89b4fa')
        ax.annotate('', xy=(8.2,2), xytext=(1.8,5), arrowprops=dict(arrowstyle='->', color='#f38ba8', lw=1.5, linestyle='dashed'))
        ax.text(4.5, 3.9, f'p = {p:.3f}', ha='center', fontsize=9, color='#f38ba8', rotation=-30)
        ax.annotate('', xy=(8.2,5), xytext=(1.8,2), arrowprops=dict(arrowstyle='->', color='#f38ba8', lw=1.5, linestyle='dashed'))
        ax.text(4.5, 3.3, f'p = {p:.3f}', ha='center', fontsize=9, color='#f38ba8', rotation=30)
        ax.text(0,6,'Input X', fontsize=11, color='#7f849c', fontweight='bold')
        ax.text(8,6,'Output Y', fontsize=11, color='#7f849c', fontweight='bold')
        ax.set_title('Binary Symmetric Channel', fontsize=12, pad=5)
        self.bsc_canvas.fig.tight_layout(); self.bsc_canvas.draw()

        info = [f"{'═'*36}\n  BSC Channel Capacity\n{'═'*36}\n\n"]
        info.append(f"Crossover probability p = {p:.3f}\n")
        info.append(f"Binary entropy H(p) = {hp:.4f} bits\n\n")
        info.append(f"C = 1 − H(p) = 1 − {hp:.4f} = {cap:.4f}\n\n")
        if p < 0.001:
            info.append("✓ No noise → C ≈ 1 bit/use (maximum)")
        elif abs(p-0.5) < 0.01:
            info.append("✗ p ≈ 0.5 → C ≈ 0 (useless channel)")
        elif p > 0.5:
            info.append("⚡ p > 0.5: flip all outputs to recover info!")
        else:
            info.append(f"✓ Reliable comm. at rate ≤ {cap:.4f} bits/use")
        self.ch_info.setText(''.join(info))

        ax2 = self.cap_canvas.axes; ax2.clear()
        x = np.linspace(0, 1, 1000)
        y = np.array([1 - binary_entropy(xi) for xi in x])
        ax2.plot(x, y, color='#89b4fa', linewidth=2.5, label='C = 1 − H(p)')
        ax2.fill_between(x, y, alpha=0.12, color='#89b4fa')
        ax2.axvline(x=p, color='#fab387', linestyle='--', alpha=0.7)
        ax2.plot(p, cap, 'o', color='#fab387', markersize=10, zorder=5,
                 label=f'p={p:.3f}, C={cap:.3f}')
        ax2.set_xlabel('Crossover Probability p'); ax2.set_ylabel('Capacity [bits/use]')
        ax2.set_title('BSC Capacity vs. Crossover Probability')
        ax2.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=9)
        ax2.set_xlim(0,1); ax2.set_ylim(-0.05, 1.1)
        ax2.grid(True, alpha=0.15, color='#3a3a55')
        self.cap_canvas.fig.tight_layout(); self.cap_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — KL DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════

class KLDivergencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("KL Divergence & Cross Entropy"))
        lay.addWidget(formula_label(
            "D_KL(P‖Q) = Σ P(x)·log₂(P(x)/Q(x))\n"
            "H(P,Q) = −Σ P(x)·log₂ Q(x) = D_KL(P‖Q) + H(P)"
        ))
        lay.addWidget(info_label(
            "KL divergence measures how distribution Q diverges from reference P. "
            "It is NOT symmetric: D_KL(P‖Q) ≠ D_KL(Q‖P). Cross entropy H(P,Q) "
            "counts the bits needed when encoding data from P using a code optimized for Q."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        ig = QGroupBox("Distributions")
        il = QVBoxLayout(ig)
        pl = QLabel("Distribution P (reference):")
        pl.setStyleSheet("color:#89b4fa; font-weight:bold;"); il.addWidget(pl)
        self.p_input = QLineEdit("0.4, 0.3, 0.2, 0.1"); il.addWidget(self.p_input)
        ql = QLabel("Distribution Q (approximation):")
        ql.setStyleSheet("color:#fab387; font-weight:bold;"); il.addWidget(ql)
        self.q_input = QLineEdit("0.25, 0.25, 0.25, 0.25"); il.addWidget(self.q_input)
        btn = QPushButton("Calculate"); btn.clicked.connect(self._calc); il.addWidget(btn)

        prl = QHBoxLayout()
        for name, pv, qv in [("P=Q", "0.4,0.3,0.2,0.1", "0.4,0.3,0.2,0.1"),
                              ("P uniform", "0.25,0.25,0.25,0.25", "0.4,0.3,0.2,0.1"),
                              ("P skewed", "0.8,0.1,0.05,0.05", "0.25,0.25,0.25,0.25")]:
            b = QPushButton(name); b.setObjectName("secondary")
            b.clicked.connect(lambda _, a=pv, c=qv: (self.p_input.setText(a),
                            self.q_input.setText(c), self._calc()))
            prl.addWidget(b)
        il.addLayout(prl)

        self.kl_lbl = result_label("D_KL(P‖Q) = --")
        il.addWidget(self.kl_lbl)
        self.kl_rev = QLabel("D_KL(Q‖P) = --")
        self.kl_rev.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#fab387;")
        il.addWidget(self.kl_rev)
        self.ce_lbl = QLabel("H(P,Q) = --")
        self.ce_lbl.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#f38ba8;")
        il.addWidget(self.ce_lbl)

        self.kl_detail = detail_textedit()
        self.kl_detail.setMaximumHeight(180)
        il.addWidget(self.kl_detail)
        ll.addWidget(ig)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.kl_canvas = MplCanvas(self, width=6, height=5)
        rl.addWidget(self.kl_canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._calc()

    def _calc(self):
        try:
            p = np.array([float(x.strip()) for x in self.p_input.text().split(',') if x.strip()])
            q = np.array([float(x.strip()) for x in self.q_input.text().split(',') if x.strip()])
            if len(p) != len(q):
                self.kl_lbl.setText("⚠  P and Q must have same length"); return
            if abs(p.sum() - 1.0) > 0.01 or abs(q.sum() - 1.0) > 0.01:
                self.kl_lbl.setText("⚠  Both must sum to 1.0"); return
            if any(x <= 0 for x in q):
                self.kl_lbl.setText("⚠  Q must have all positive values"); return

            hp = shannon_entropy(p)
            dkl = kl_divergence(p, q)
            dkl_rev = kl_divergence(q, p)
            ce = cross_entropy(p, q)

            self.kl_lbl.setText(f"D_KL(P‖Q) = {dkl:.4f} bits")
            self.kl_rev.setText(f"D_KL(Q‖P) = {dkl_rev:.4f} bits   (asymmetric!)")
            self.ce_lbl.setText(f"H(P,Q)    = {ce:.4f} bits = H(P)+D_KL = {hp:.4f}+{dkl:.4f}")

            txt = [f"{'═'*38}\n  KL Divergence Step-by-Step\n{'═'*38}\n\n"]
            txt.append(f"P = [{', '.join(f'{x:.3f}' for x in p)}]\n")
            txt.append(f"Q = [{', '.join(f'{x:.3f}' for x in q)}]\n\n")
            txt.append(f"{'─'*38}\n  D_KL(P‖Q) terms\n{'─'*38}\n")
            for i in range(len(p)):
                if p[i] > 0 and q[i] > 0:
                    t = p[i] * np.log2(p[i] / q[i])
                    txt.append(f"  P(x{i})·log₂(P(x{i})/Q(x{i})) = "
                               f"{p[i]:.3f}·log₂({p[i]:.3f}/{q[i]:.3f}) = {t:.4f}\n")
            txt.append(f"\n  D_KL(P‖Q) = {dkl:.4f} bits\n\n")
            txt.append(f"{'─'*38}\n  Key Properties\n{'─'*38}\n")
            txt.append(f"  D_KL(P‖Q) = {dkl:.4f}  ≠  D_KL(Q‖P) = {dkl_rev:.4f}\n")
            txt.append(f"  ⇒ KL divergence is NOT symmetric\n\n")
            txt.append(f"  H(P)   = {hp:.4f}  (entropy of P)\n")
            txt.append(f"  H(P,Q) = {ce:.4f}  (cross-entropy)\n")
            txt.append(f"  H(P,Q) = H(P) + D_KL(P‖Q)\n")
            txt.append(f"         = {hp:.4f} + {dkl:.4f} = {hp+dkl:.4f} ✓\n")
            if dkl < 0.001:
                txt.append(f"\n  ✓ P ≈ Q → D_KL ≈ 0")
            self.kl_detail.setText(''.join(txt))

            ax = self.kl_canvas.axes; ax.clear()
            n = len(p); syms = [f'x{i}' for i in range(n)]
            bw = 0.35; xs = np.arange(n)
            ax.bar(xs - bw/2, p, bw, label='P', color='#89b4fa', alpha=0.85, edgecolor='#ffffff10')
            ax.bar(xs + bw/2, q, bw, label='Q', color='#fab387', alpha=0.85, edgecolor='#ffffff10')
            ax.set_xticks(xs); ax.set_xticklabels(syms)
            ax.set_xlabel('Symbol'); ax.set_ylabel('Probability')
            ax.set_title(f'P vs Q   |   D_KL(P‖Q) = {dkl:.4f}')
            ax.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=10)
            ax.set_ylim(0, max(max(p), max(q)) * 1.3)
            ax.grid(True, alpha=0.15, color='#3a3a55', axis='y')
            self.kl_canvas.fig.tight_layout(); self.kl_canvas.draw()
        except (ValueError, ZeroDivisionError) as e:
            self.kl_lbl.setText(f"⚠  Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — SOURCE CODING THEOREM
# ═══════════════════════════════════════════════════════════════════════════════

class SourceCodingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Source Coding Theorem"))
        lay.addWidget(formula_label(
            "H(X) ≤ L̄ < H(X) + 1   [bits per symbol]\n"
            "For N-block coding:  H(X) ≤ L̄_N < H(X) + 1/N"
        ))
        lay.addWidget(info_label(
            "Shannon's Source Coding Theorem establishes that the entropy H(X) is the "
            "fundamental limit on lossless compression. No code can achieve average length "
            "less than H(X). Huffman coding achieves L̄ < H(X) + 1 for single symbols. "
            "Block coding (encoding N symbols together) approaches H(X) as N → ∞."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        ig = QGroupBox("Source Distribution")
        il = QVBoxLayout(ig)
        nl = QLabel("Number of symbols:")
        nl.setStyleSheet("color:#7f849c; font-size:12px;"); il.addWidget(nl)
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(2, 10); self.n_slider.setValue(4)
        self.n_slider.valueChanged.connect(self._update)
        self.n_label = QLabel("n = 4")
        self.n_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px;")
        il.addWidget(self.n_label); il.addWidget(self.n_slider)

        sl = QLabel("Skewness (0 = uniform, 1 = very skewed):")
        sl.setStyleSheet("color:#7f849c; font-size:12px;"); il.addWidget(sl)
        self.skew_slider = QSlider(Qt.Horizontal)
        self.skew_slider.setRange(0, 100); self.skew_slider.setValue(50)
        self.skew_slider.valueChanged.connect(self._update)
        self.skew_label = QLabel("skew = 0.50")
        self.skew_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px;")
        il.addWidget(self.skew_label); il.addWidget(self.skew_slider)
        ll.addWidget(ig)

        rg = QGroupBox("Theorem Verification")
        rl = QVBoxLayout(rg)
        self.sc_h = result_label("H(X) = --")
        self.sc_lavg = QLabel("L̄ (Huffman) = --")
        self.sc_lavg.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#fab387; font-weight:bold;")
        self.sc_check = QLabel("")
        self.sc_check.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#a6e3a1; padding:8px;")
        self.sc_detail = detail_textedit()
        rl.addWidget(self.sc_h); rl.addWidget(self.sc_lavg); rl.addWidget(self.sc_check)
        rl.addWidget(self.sc_detail)
        ll.addWidget(rg)
        content.addWidget(left, 2)

        right = QWidget(); rl2 = QVBoxLayout(right)
        self.sc_canvas = MplCanvas(self, width=6, height=5)
        rl2.addWidget(self.sc_canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        n = self.n_slider.value()
        skew = self.skew_slider.value() / 100.0
        self.n_label.setText(f"n = {n}")
        self.skew_label.setText(f"skew = {skew:.2f}")

        if skew < 0.01:
            probs = np.ones(n) / n
        else:
            alpha = 0.5 + skew * 5
            raw = np.array([alpha ** i for i in range(n)])
            probs = raw / raw.sum()

        symbols = [f"s{i}" for i in range(n)]
        h = shannon_entropy(probs)
        sp = list(zip(symbols, probs.tolist()))
        root, _ = build_huffman_tree(sp)
        codes = get_huffman_codes(root)
        l_avg = sum(probs[i] * len(codes[symbols[i]]) for i in range(n) if symbols[i] in codes)
        hmax = math.log2(n)

        self.sc_h.setText(f"H(X)       = {h:.4f} bits")
        self.sc_lavg.setText(f"L̄ (Huffman)= {l_avg:.4f} bits")

        if l_avg >= h - 1e-10 and l_avg < h + 1 + 1e-10:
            check = f"✓  H(X) ≤ L̄ < H(X)+1  →  {h:.4f} ≤ {l_avg:.4f} < {h+1:.4f}"
        else:
            check = f"✗  Theorem check: {h:.4f} ≤ {l_avg:.4f} < {h+1:.4f}"
        self.sc_check.setText(check)

        txt = [f"{'═'*38}\n  Source Coding Theorem\n{'═'*38}\n\n"]
        txt.append(f"Distribution: [{', '.join(f'{p:.4f}' for p in probs)}]\n\n")
        txt.append(f"{'─'*38}\n  Single Symbol Coding\n{'─'*38}\n")
        txt.append(f"  Entropy      H(X) = {h:.4f} bits\n")
        txt.append(f"  Avg length   L̄    = {l_avg:.4f} bits\n")
        txt.append(f"  Redundancy   L̄−H  = {l_avg-h:.4f} bits\n")
        txt.append(f"  Efficiency        = {h/l_avg*100:.1f}%\n")
        txt.append(f"  Max entropy       = {hmax:.4f} bits (uniform)\n\n")
        txt.append(f"{'─'*38}\n  Huffman Codes\n{'─'*38}\n")
        for i, s in enumerate(symbols):
            c = codes.get(s, "?")
            txt.append(f"  {s}  p={probs[i]:.4f}  →  {c}  (len={len(c)})\n")
        txt.append(f"\n{'─'*38}\n  Block Coding (N symbols)\n{'─'*38}\n")
        for block_n in [1, 2, 4, 8]:
            block_h = block_n * h
            bound = block_h + (1.0 / block_n)
            txt.append(f"  N={block_n}: H_block={block_h:.3f}, "
                       f"upper bound L̄_N < {bound:.3f} "
                       f"(overhead < {1.0/block_n:.3f})\n")
        txt.append(f"\n  As N → ∞:  L̄_N → H(X) = {h:.4f}\n")
        self.sc_detail.setText(''.join(txt))

        ax = self.sc_canvas.axes; ax.clear()
        xs = np.arange(n)
        colors = ['#89b4fa', '#fab387', '#a6e3a1', '#f38ba8', '#cba6f7',
                  '#94e2d5', '#f9e2af', '#74c7ec', '#eba0ac', '#89dceb']
        ax.bar(xs, probs, color=colors[:n], alpha=0.85, edgecolor='#ffffff10')
        for i, s in enumerate(symbols):
            if s in codes:
                ax.text(i, probs[i] + max(probs)*0.05, f'len={len(codes[s])}',
                        ha='center', fontsize=9, color='#f9e2af', fontweight='bold')
        ax.axhline(y=1/n, color='#585b70', linestyle=':', alpha=0.5, label='Uniform (1/n)')
        ax.text(0.98, 0.95, f'H(X) = {h:.3f}\nL̄ = {l_avg:.3f}\nη = {h/l_avg*100:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#a6e3a1',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a3a2a', edgecolor='#a6e3a1', alpha=0.8))
        ax.set_xlabel('Symbol'); ax.set_ylabel('Probability')
        ax.set_title('Source Distribution & Huffman Code Lengths')
        ax.set_xticks(xs); ax.set_xticklabels(symbols)
        ax.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=9)
        ax.set_ylim(0, max(probs) * 1.4 if max(probs) > 0 else 1)
        ax.grid(True, alpha=0.15, color='#3a3a55', axis='y')
        self.sc_canvas.fig.tight_layout(); self.sc_canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 9 — PRACTICE QUIZ
# ═══════════════════════════════════════════════════════════════════════════════

class QuizPage(QWidget):
    Q_TEMPLATES = [
        {"topic": "Entropy", "q": "A fair coin has entropy H = ? (in bits)",
         "answer": "1", "hint": "H = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1"},
        {"topic": "Entropy", "q": "A fair 8-sided die has entropy H = ? (2 d.p.)",
         "answer_fn": lambda: f"{math.log2(8):.2f}", "hint": "H = log₂(8) = 3"},
        {"topic": "Binary Entropy", "q": "What is H(0.1)? (4 d.p.)",
         "answer_fn": lambda: f"{binary_entropy(0.1):.4f}",
         "hint": "H(0.1) = -0.1·log₂(0.1) - 0.9·log₂(0.9)"},
        {"topic": "Binary Entropy", "q": "What is the maximum value of the binary entropy function?",
         "answer": "1", "hint": "Maximum at p=0.5: H(0.5)=1 bit"},
        {"topic": "Mutual Information", "q": "If X and Y are independent, what is I(X;Y)?",
         "answer": "0", "hint": "Independence ⟹ H(X,Y)=H(X)+H(Y) ⟹ I=0"},
        {"topic": "KL Divergence", "q": "What is D_KL(P‖P)? (divergence of P from itself)",
         "answer": "0", "hint": "D_KL(P‖P) = Σ P(x)·log(P(x)/P(x)) = 0"},
        {"topic": "Channel Capacity", "q": "For a noiseless BSC (p=0), what is the capacity C?",
         "answer": "1", "hint": "C = 1 - H(0) = 1 - 0 = 1 bit/use"},
        {"topic": "Channel Capacity", "q": "For a BSC with p=0.5, what is the capacity C?",
         "answer": "0", "hint": "C = 1 - H(0.5) = 1 - 1 = 0 bits/use"},
        {"topic": "Source Coding", "q": "Can a lossless code achieve avg length < H(X)? (yes/no)",
         "answer": "no", "hint": "Source Coding Theorem: H(X) is the lower bound"},
        {"topic": "Entropy", "q": "A uniform distribution over n outcomes has entropy = ? (in terms of n)",
         "answer": "log2(n)", "alt": ["log₂(n)", "log2 n", "log(n)/log(2)", "log n base 2"],
         "hint": "H = -Σ (1/n)·log₂(1/n) = log₂(n)"},
        {"topic": "Fano", "q": "Fano's inequality gives a lower bound on what quantity?",
         "answer": "error probability", "alt": ["pe", "p_e", "probability of error"],
         "hint": "P_e ≥ (H(X|Y)-1)/log₂(|X̂|)"},
        {"topic": "Hamming", "q": "Hamming(7,4) can correct up to how many bit errors?",
         "answer": "1", "hint": "Minimum distance = 3, so t = floor((3-1)/2) = 1"},
        {"topic": "Joint Entropy", "q": "If X and Y are independent, H(X,Y) = ?",
         "answer": "H(X)+H(Y)", "alt": ["h(x)+h(y)"], "hint": "Independence ⟹ H(X,Y) = H(X)+H(Y)"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Practice Quiz"))
        lay.addWidget(info_label(
            "Test your understanding of information theory concepts. "
            "Type your answer and press Check. Numeric answers accept ±0.02 tolerance."
        ))

        score_lay = QHBoxLayout()
        self.score_label = QLabel("Score: 0 / 0")
        self.score_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #f9e2af; font-family:'Consolas',monospace;")
        score_lay.addWidget(self.score_label)
        score_lay.addStretch()
        self.streak_label = QLabel("Streak: 0")
        self.streak_label.setStyleSheet("font-size: 14px; color: #fab387; font-family:'Consolas',monospace;")
        score_lay.addWidget(self.streak_label)
        lay.addLayout(score_lay)

        q_group = QGroupBox("Question")
        q_lay = QVBoxLayout(q_group)
        self.topic_badge = QLabel("Topic")
        self.topic_badge.setStyleSheet(
            "background-color: #2d2d50; color: #89b4fa; padding: 4px 12px; "
            "border-radius: 10px; font-size: 12px; font-weight: bold;")
        q_lay.addWidget(self.topic_badge)
        self.question_label = QLabel("Loading...")
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("font-size: 18px; color: #cdd6f4; padding: 15px 0;")
        q_lay.addWidget(self.question_label)

        ans_lay = QHBoxLayout()
        self.answer_input = QLineEdit()
        self.answer_input.setPlaceholderText("Type your answer here...")
        self.answer_input.returnPressed.connect(self._check_answer)
        ans_lay.addWidget(self.answer_input, 3)
        check_btn = QPushButton("Check"); check_btn.clicked.connect(self._check_answer)
        ans_lay.addWidget(check_btn, 1)
        skip_btn = QPushButton("Skip"); skip_btn.setObjectName("secondary")
        skip_btn.clicked.connect(self._next_question)
        ans_lay.addWidget(skip_btn, 1)
        q_lay.addLayout(ans_lay)

        self.feedback_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("font-size: 14px; padding: 8px;")
        q_lay.addWidget(self.feedback_label)
        hint_btn = QPushButton("Show Hint"); hint_btn.setObjectName("secondary")
        hint_btn.clicked.connect(self._show_hint)
        q_lay.addWidget(hint_btn)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet("color: #f9e2af; font-style: italic; font-size: 12px; padding: 5px;")
        self.hint_label.hide()
        q_lay.addWidget(self.hint_label)
        lay.addWidget(q_group)

        hist_group = QGroupBox("Answer History")
        hist_lay = QVBoxLayout(hist_group)
        self.history_text = detail_textedit()
        self.history_text.setMaximumHeight(200)
        hist_lay.addWidget(self.history_text)
        lay.addWidget(hist_group)
        lay.addStretch()

        self.score = 0; self.total = 0; self.streak = 0
        self.current_q = None; self.history_lines = []
        self._next_question()

    def _next_question(self):
        self.current_q = random.choice(self.Q_TEMPLATES)
        self.topic_badge.setText(self.current_q["topic"])
        self.question_label.setText(self.current_q["q"])
        self.answer_input.clear()
        self.feedback_label.setText("")
        self.hint_label.setText(""); self.hint_label.hide()
        self.answer_input.setFocus()

    def _get_correct_answer(self):
        q = self.current_q
        if "answer_fn" in q:
            return q["answer_fn"]()
        return q.get("answer", "")

    def _check_answer(self):
        if self.current_q is None: return
        user = self.answer_input.text().strip().lower()
        correct = self._get_correct_answer().strip().lower()
        alt_answers = [a.lower() for a in self.current_q.get("alt", [])] + [correct]

        is_correct = False
        try:
            is_correct = abs(float(user) - float(correct)) < 0.02
        except ValueError:
            is_correct = user in alt_answers

        self.total += 1
        if is_correct:
            self.score += 1; self.streak += 1
            self.feedback_label.setText("✅  Correct!")
            self.feedback_label.setStyleSheet("font-size:14px; color:#a6e3a1; padding:8px; font-weight:bold;")
            self.history_lines.append(f"  ✅ {self.current_q['q']}\n     Answer: {user}  ✓\n")
        else:
            self.streak = 0
            self.feedback_label.setText(f"❌  Incorrect. Correct answer: {correct}")
            self.feedback_label.setStyleSheet("font-size:14px; color:#f38ba8; padding:8px; font-weight:bold;")
            self.history_lines.append(f"  ❌ {self.current_q['q']}\n     Yours: {user}  |  Correct: {correct}\n")

        self.score_label.setText(f"Score: {self.score} / {self.total}")
        fire = " 🔥" if self.streak >= 3 else ""
        self.streak_label.setText(f"Streak: {self.streak}{fire}")
        self.history_text.setText('\n'.join(self.history_lines[-20:]))
        QTimer.singleShot(1500, self._next_question)

    def _show_hint(self):
        if self.current_q:
            self.hint_label.setText(f"💡 Hint: {self.current_q.get('hint', 'No hint available.')}")
            self.hint_label.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 10 — NOISY CHANNEL CODING THEOREM
# ═══════════════════════════════════════════════════════════════════════════════

class NoisyChannelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Noisy Channel Coding Theorem"))
        lay.addWidget(formula_label(
            "For rate R < C:  ∃ code with P_e → 0 as n → ∞\n"
            "For rate R > C:  P_e is bounded away from 0"
        ))
        lay.addWidget(info_label(
            "Shannon's second theorem (1948) is perhaps the most surprising result in information theory: "
            "even over a noisy channel, arbitrarily reliable communication is possible as long as the "
            "transmission rate R is below the channel capacity C."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)
        cg = QGroupBox("Parameters")
        cl = QVBoxLayout(cg)
        cl.addWidget(QLabel("Crossover probability p:"))
        self.p_slider = QSlider(Qt.Horizontal)
        self.p_slider.setRange(1, 499); self.p_slider.setValue(110)
        self.p_slider.valueChanged.connect(self._update)
        self.p_label = QLabel("p = 0.110")
        self.p_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px;")
        cl.addWidget(self.p_label); cl.addWidget(self.p_slider)

        cl.addWidget(QLabel("Code rate R (bits/use):"))
        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setRange(1, 99); self.r_slider.setValue(40)
        self.r_slider.valueChanged.connect(self._update)
        self.r_label = QLabel("R = 0.40")
        self.r_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px;")
        cl.addWidget(self.r_label); cl.addWidget(self.r_slider)

        self.status_label = result_label("R < C ✓")
        cl.addWidget(self.status_label)
        ll.addWidget(cg)

        ig = QGroupBox("Theorem Details")
        il = QVBoxLayout(ig)
        self.detail_text = detail_textedit()
        il.addWidget(self.detail_text)
        ll.addWidget(ig)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.canvas = MplCanvas(self, width=6, height=6)
        rl.addWidget(self.canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        p = self.p_slider.value() / 1000.0
        R = self.r_slider.value() / 100.0
        self.p_label.setText(f"p = {p:.3f}")
        self.r_label.setText(f"R = {R:.2f}")
        C = 1 - binary_entropy(p)

        ax = self.canvas.axes; ax.clear()
        n_vals = np.logspace(1, 6, 500)

        def gallager_e0(rho, p_val):
            if rho <= 0: return 0.0
            term = (p_val ** (1/(1+rho)) + (1-p_val) ** (1/(1+rho))) ** (1+rho)
            return -np.log2(max(term, 1e-30))

        rhos = np.linspace(0.001, 1.0, 200)
        E_R = 0.0
        for rho in rhos:
            e0 = gallager_e0(rho, p)
            e = e0 - rho * R
            if e > E_R: E_R = e

        if R < C and E_R > 0:
            pe_bound = np.minimum(2.0 ** (-n_vals * E_R), 1.0)
            ax.loglog(n_vals, pe_bound, color='#a6e3a1', linewidth=2.5,
                      label=f'P_e bound (R={R:.2f} < C={C:.2f})')
            self.status_label.setText(f"R = {R:.2f} < C = {C:.4f} ✓\nReliable comm. possible")
            self.status_label.setStyleSheet(
                "font-family:'Consolas',monospace; font-size:14px; color:#a6e3a1; font-weight:bold;")
        else:
            pe_lower = max(1 - C / R, 0) if R > 0 else 0
            ax.axhline(y=max(pe_lower, 1e-6), color='#f38ba8', linewidth=2.5, linestyle='--',
                        label=f'P_e ≥ {pe_lower:.3f}  (R={R:.2f} > C={C:.2f})')
            ax.set_ylim(1e-7, 2)
            self.status_label.setText(f"R = {R:.2f} > C = {C:.4f} ✗\nReliable comm. IMPOSSIBLE")
            self.status_label.setStyleSheet(
                "font-family:'Consolas',monospace; font-size:14px; color:#f38ba8; font-weight:bold;")

        ax.axhline(y=1e-3, color='#585b70', linestyle=':', alpha=0.3)
        ax.set_xlabel('Blocklength n'); ax.set_ylabel('Error Probability P_e')
        ax.set_title(f'Noisy Channel Coding: BSC(p={p:.3f}), C={C:.3f}, R={R:.2f}')
        ax.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.15, color='#3a3a55', which='both')
        ax.set_xlim(10, 1e6)
        if R < C: ax.set_ylim(1e-30, 2)
        self.canvas.fig.tight_layout(); self.canvas.draw()

        txt = [f"{'═'*40}\n  Noisy Channel Coding Theorem\n{'═'*40}\n\n"]
        txt.append(f"Channel: BSC,  p = {p:.3f},  C = {C:.4f}\n")
        txt.append(f"Code rate: R = {R:.2f}\n\n")
        if R < C:
            txt.append(f"✓ R < C: Reliable communication IS possible\n\n")
            txt.append(f"  Gallager's E(R) ≈ {E_R:.4f}\n")
            txt.append(f"  P_e ≤ 2^(-n·E(R)) = 2^(-n·{E_R:.4f})\n\n")
            for n_test in [100, 1000, 10000, 100000]:
                pe_est = min(2.0 ** (-n_test * E_R), 1.0)
                txt.append(f"  n = {n_test:>6d}  →  P_e ≤ {pe_est:.2e}\n")
        else:
            pe_lb = max(1 - C / R, 0)
            txt.append(f"✗ R > C: Reliable communication is NOT possible\n\n")
            txt.append(f"  P_e ≥ 1 - C/R = 1 - {C:.4f}/{R:.2f} = {pe_lb:.4f}\n")
        self.detail_text.setText(''.join(txt))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 11 — FANO'S INEQUALITY
# ═══════════════════════════════════════════════════════════════════════════════

class FanoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Fano's Inequality"))
        lay.addWidget(formula_label(
            "H(P_e) + P_e · log₂(|X̂|−1)  ≥  H(X|Y)\n\n"
            "P_e  ≥  (H(X|Y) − 1) / log₂(|X̂|)"
        ))
        lay.addWidget(info_label(
            "Fano's inequality connects the conditional entropy H(X|Y) to the minimum "
            "error probability P_e when guessing X from Y. It is fundamental in proving "
            "converse results in information theory."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)
        cg = QGroupBox("Parameters")
        cl = QVBoxLayout(cg)
        cl.addWidget(QLabel("Conditional entropy H(X|Y):"))
        self.hxy_slider = QSlider(Qt.Horizontal)
        self.hxy_slider.setRange(0, 300); self.hxy_slider.setValue(150)
        self.hxy_slider.valueChanged.connect(self._update)
        self.hxy_label = QLabel("H(X|Y) = 1.500 bits")
        self.hxy_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px;")
        cl.addWidget(self.hxy_label); cl.addWidget(self.hxy_slider)

        cl.addWidget(QLabel("Number of symbols |X̂|:"))
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(2, 32); self.n_slider.setValue(4)
        self.n_slider.valueChanged.connect(self._update)
        self.n_label = QLabel("|X̂| = 4")
        self.n_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px;")
        cl.addWidget(self.n_label); cl.addWidget(self.n_slider)

        self.pe_label = result_label("P_e ≥ 0.3333")
        cl.addWidget(self.pe_label)
        ll.addWidget(cg)

        sg = QGroupBox("Step-by-Step Derivation")
        sl = QVBoxLayout(sg)
        self.fano_detail = detail_textedit()
        sl.addWidget(self.fano_detail)
        ll.addWidget(sg)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.canvas = MplCanvas(self, width=6, height=5)
        rl.addWidget(self.canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _update(self):
        hxy = self.hxy_slider.value() / 100.0
        n_sym = self.n_slider.value()
        self.hxy_label.setText(f"H(X|Y) = {hxy:.2f} bits")
        self.n_label.setText(f"|X̂| = {n_sym}")

        pe_lower = max((hxy - 1) / math.log2(n_sym), 0) if n_sym > 1 else 0
        self.pe_label.setText(f"P_e ≥ {pe_lower:.4f}")

        txt = [f"{'═'*40}\n  Fano's Inequality — Step by Step\n{'═'*40}\n\n"]
        txt.append(f"Given:\n  H(X|Y) = {hxy:.2f} bits\n  |X̂|    = {n_sym} symbols\n\n")
        txt.append(f"{'─'*40}\n  Simplified Lower Bound\n{'─'*40}\n\n")
        txt.append(f"  Since H(P_e) ≤ 1 for any P_e:\n\n")
        txt.append(f"  P_e · log₂(|X̂|−1) ≥ H(X|Y) − H(P_e)\n")
        txt.append(f"                      ≥ H(X|Y) − 1\n\n")
        txt.append(f"  ∴  P_e ≥ (H(X|Y) − 1) / log₂(|X̂|)\n")
        txt.append(f"         = ({hxy:.2f} − 1) / log₂({n_sym})\n")
        txt.append(f"         = {max(hxy-1,0):.2f} / {math.log2(n_sym):.4f}\n")
        txt.append(f"         = {pe_lower:.4f}\n\n")
        if hxy < 0.01:
            txt.append(f"  ✓ H(X|Y) ≈ 0: Y almost completely determines X\n")
        elif hxy > math.log2(n_sym) - 0.1:
            txt.append(f"  ✗ H(X|Y) ≈ log₂(|X̂|): Y provides almost no info\n")
        else:
            txt.append(f"  → Minimum error probability: P_e ≥ {pe_lower:.4f}\n")
        self.fano_detail.setText(''.join(txt))

        ax = self.canvas.axes; ax.clear()
        pe_vals = np.linspace(0.001, 0.999, 500)
        fano_vals = np.array([binary_entropy(pe) + pe * math.log2(max(n_sym-1, 1)) for pe in pe_vals])
        ax.plot(fano_vals, pe_vals, color='#89b4fa', linewidth=2.5,
                label="H(Pe)+Pe·log₂(|X̂|−1) vs Pe")
        ax.axvline(x=hxy, color='#fab387', linestyle='--', linewidth=2, label=f'H(X|Y) = {hxy:.2f}')
        ax.axhline(y=pe_lower, color='#a6e3a1', linestyle=':', linewidth=1.5, label=f'P_e ≥ {pe_lower:.4f}')
        ax.axhspan(0, pe_lower, alpha=0.08, color='#f38ba8')
        if pe_lower > 0.01:
            ax.text(hxy + 0.05, pe_lower / 2, 'Impossible\nregion', fontsize=10, color='#f38ba8', style='italic')
        ax.set_xlabel('H(X|Y)  [bits]'); ax.set_ylabel('Error Probability P_e')
        ax.set_title(f"Fano's Inequality  (|X̂| = {n_sym})")
        ax.legend(facecolor='#1e1e35', edgecolor='#2a2a40', labelcolor='#a6adc8', fontsize=9, loc='upper left')
        ax.set_xlim(0, math.log2(n_sym) + 0.5); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.15, color='#3a3a55')
        self.canvas.fig.tight_layout(); self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 12 — HAMMING CODE
# ═══════════════════════════════════════════════════════════════════════════════

class HammingCodePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Hamming(7,4) Code"))
        lay.addWidget(formula_label(
            "Encode:  [d₁d₂d₃d₄] → [p₁p₂d₁p₃d₂d₃d₄]\n"
            "Decode:  Syndrome → error position → correct"
        ))
        lay.addWidget(info_label(
            "Hamming(7,4) encodes 4 data bits into 7 bits by adding 3 parity bits. "
            "It can detect and correct any single-bit error. Enter 4 data bits to encode, "
            "then flip any bit to see the decoder find and fix the error."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        eg = QGroupBox("Step 1: Encode")
        el = QVBoxLayout(eg)
        el.addWidget(QLabel("Enter 4 data bits (0/1):"))
        self.data_input = QLineEdit("1011")
        self.data_input.setMaxLength(4)
        el.addWidget(self.data_input)
        enc_btn = QPushButton("Encode"); enc_btn.clicked.connect(self._encode)
        el.addWidget(enc_btn)
        self.encoded_label = result_label("Codeword: —")
        el.addWidget(self.encoded_label)
        self.parity_detail = detail_textedit()
        self.parity_detail.setMaximumHeight(140)
        el.addWidget(self.parity_detail)
        ll.addWidget(eg)

        fg = QGroupBox("Step 2: Introduce Error")
        fl = QVBoxLayout(fg)
        fl.addWidget(QLabel("Flip bit at position (0 = no error):"))
        self.flip_slider = QSlider(Qt.Horizontal)
        self.flip_slider.setRange(0, 7); self.flip_slider.setValue(0)
        self.flip_slider.valueChanged.connect(self._decode)
        self.flip_info = QLabel("No error")
        self.flip_info.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#a6e3a1;")
        fl.addWidget(self.flip_info); fl.addWidget(self.flip_slider)
        self.received_label = QLabel("Received: —")
        self.received_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#f38ba8; font-weight:bold;")
        fl.addWidget(self.received_label)
        ll.addWidget(fg)

        dg = QGroupBox("Step 3: Decode & Correct")
        dl = QVBoxLayout(dg)
        self.syndrome_label = result_label("Syndrome: —")
        dl.addWidget(self.syndrome_label)
        self.error_pos_label = QLabel("Error position: —")
        self.error_pos_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#fab387; font-weight:bold;")
        dl.addWidget(self.error_pos_label)
        self.corrected_label = QLabel("Corrected: —")
        self.corrected_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#a6e3a1; font-weight:bold;")
        dl.addWidget(self.corrected_label)
        self.decode_detail = detail_textedit()
        self.decode_detail.setMaximumHeight(160)
        dl.addWidget(self.decode_detail)
        ll.addWidget(dg)
        content.addWidget(left, 2)

        right = QWidget(); rl = QVBoxLayout(right)
        self.canvas = MplCanvas(self, width=6, height=6)
        rl.addWidget(self.canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self.codeword = [0] * 7
        self._encode()

    def _encode(self):
        data_str = self.data_input.text().strip()
        if len(data_str) != 4 or not all(c in '01' for c in data_str):
            self.encoded_label.setText("⚠  Enter exactly 4 binary digits"); return
        d = [int(c) for c in data_str]
        p1 = d[0] ^ d[1] ^ d[3]
        p2 = d[0] ^ d[2] ^ d[3]
        p3 = d[1] ^ d[2] ^ d[3]
        self.codeword = [p1, p2, d[0], p3, d[1], d[2], d[3]]
        self.encoded_label.setText(f"Codeword: [{''.join(str(b) for b in self.codeword)}]")

        txt = [f"Data: d₁={d[0]} d₂={d[1]} d₃={d[2]} d₄={d[3]}\n\n"]
        txt.append(f"p₁ = d₁⊕d₂⊕d₄ = {d[0]}⊕{d[1]}⊕{d[3]} = {p1}\n")
        txt.append(f"p₂ = d₁⊕d₃⊕d₄ = {d[0]}⊕{d[2]}⊕{d[3]} = {p2}\n")
        txt.append(f"p₃ = d₂⊕d₃⊕d₄ = {d[1]}⊕{d[2]}⊕{d[3]} = {p3}\n\n")
        txt.append(f"Pos: 1  2  3  4  5  6  7\n")
        txt.append(f"     p₁ p₂ d₁ p₃ d₂ d₃ d₄\n")
        txt.append(f"     {p1}  {p2}  {d[0]}  {p3}  {d[1]}  {d[2]}  {d[3]}\n")
        self.parity_detail.setText(''.join(txt))
        self._decode()

    def _decode(self):
        flip_pos = self.flip_slider.value()
        received = list(self.codeword)
        if flip_pos > 0:
            received[flip_pos - 1] ^= 1

        cw_str = ''.join(str(b) for b in self.codeword)
        rx_str = ''.join(str(b) for b in received)

        if flip_pos == 0:
            self.flip_info.setText("No error introduced")
            self.received_label.setText(f"Received: [{rx_str}]  (no error)")
            self.received_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#a6e3a1; font-weight:bold;")
        else:
            self.flip_info.setText(f"Flipped bit at position {flip_pos}: {self.codeword[flip_pos-1]} → {received[flip_pos-1]}")
            self.received_label.setText(f"Received: [{rx_str}]  ← error at pos {flip_pos}")
            self.received_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#f38ba8; font-weight:bold;")

        r = received
        s1 = r[0] ^ r[2] ^ r[4] ^ r[6]
        s2 = r[1] ^ r[2] ^ r[5] ^ r[6]
        s3 = r[3] ^ r[4] ^ r[5] ^ r[6]
        syndrome = s1 * 1 + s2 * 2 + s3 * 4

        self.syndrome_label.setText(f"Syndrome: [s₁={s1}, s₂={s2}, s₃={s3}] = {syndrome}")

        corrected = list(received)
        error_detected = syndrome > 0
        if error_detected:
            corrected[syndrome - 1] ^= 1
            self.error_pos_label.setText(f"Error at position {syndrome}  →  flipping bit")
            self.error_pos_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#fab387; font-weight:bold;")
        else:
            self.error_pos_label.setText("No error detected")
            self.error_pos_label.setStyleSheet("font-family:'Consolas',monospace; font-size:14px; color:#a6e3a1; font-weight:bold;")

        corr_str = ''.join(str(b) for b in corrected)
        match = corrected == self.codeword
        self.corrected_label.setText(f"Corrected: [{corr_str}]  {'✓ CORRECT' if match else '✗ STILL WRONG'}")
        self.corrected_label.setStyleSheet(f"font-family:'Consolas',monospace; font-size:14px; font-weight:bold; color:{'#a6e3a1' if match else '#f38ba8'};")

        txt = [f"Received: [{rx_str}]\n\n"]
        txt.append(f"s₁ = r₁⊕r₃⊕r₅⊕r₇ = {r[0]}⊕{r[2]}⊕{r[4]}⊕{r[6]} = {s1}\n")
        txt.append(f"s₂ = r₂⊕r₃⊕r₆⊕r₇ = {r[1]}⊕{r[2]}⊕{r[5]}⊕{r[6]} = {s2}\n")
        txt.append(f"s₃ = r₄⊕r₅⊕r₆⊕r₇ = {r[3]}⊕{r[4]}⊕{r[5]}⊕{r[6]} = {s3}\n\n")
        txt.append(f"Syndrome = {s1}·1 + {s2}·2 + {s3}·4 = {syndrome}\n")
        if error_detected:
            txt.append(f"\n✓ Error at position {syndrome}, corrected\n")
        else:
            txt.append(f"\n✓ No errors detected\n")
        txt.append(f"Data: d₁={corrected[2]} d₂={corrected[4]} d₃={corrected[5]} d₄={corrected[6]}\n")
        self.decode_detail.setText(''.join(txt))

        # Draw
        ax = self.canvas.axes; ax.clear(); ax.axis('off')
        ax.set_xlim(-1, 9); ax.set_ylim(-1, 9)
        labels = ['p₁', 'p₂', 'd₁', 'p₃', 'd₂', 'd₃', 'd₄']
        is_parity = [True, True, False, True, False, False, False]
        ax.text(4, 8.5, 'Hamming(7,4) Bit Positions', ha='center', fontsize=14, fontweight='bold', color='#89b4fa')

        for row, (y_pos, title, bits, is_err_row) in [
            (7.2, 'Sent:', self.codeword, False),
            (5.2, 'Recv:', received, True),
            (3.0, 'Fixed:', corrected, False)
        ]:
            ax.text(0.5, y_pos, title, fontsize=11, color='#7f849c', fontweight='bold')
            for i in range(7):
                x = i + 1.2
                is_error = (flip_pos > 0 and i == flip_pos - 1 and is_err_row)
                was_corrected = (syndrome > 0 and i == syndrome - 1 and not is_err_row and flip_pos > 0)
                if is_error:
                    color, fc, lw = '#f38ba8', '#3a1a1a', 3
                elif was_corrected:
                    color, fc, lw = '#a6e3a1', '#1a3a2a', 3
                else:
                    color = '#89b4fa' if is_parity[i] else '#cba6f7'
                    fc = '#1a2040' if is_parity[i] else '#2a1a40'
                    lw = 1.5
                ax.text(x, y_pos, str(bits[i]), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=fc, edgecolor=color, linewidth=lw))
                if row == 7.2:
                    ax.text(x, y_pos - 0.6, labels[i], ha='center', fontsize=8, color='#585b70')
                if is_error:
                    ax.text(x, y_pos - 0.6, '⚠ ERR', ha='center', fontsize=8, color='#f38ba8', fontweight='bold')
                if was_corrected:
                    ax.text(x, y_pos - 0.6, '✓ FIXED', ha='center', fontsize=8, color='#a6e3a1', fontweight='bold')

        # Legend
        for x_pos, col, lbl in [(1,'#89b4fa','Parity'),(4,'#cba6f7','Data'),(7,'#f38ba8','Error')]:
            ax.text(x_pos, 0.8, '■', fontsize=16, color=col, fontweight='bold')
            ax.text(x_pos + 0.6, 0.8, lbl, fontsize=10, color='#7f849c')
        self.canvas.fig.tight_layout(); self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 13 — ARITHMETIC CODING
# ═══════════════════════════════════════════════════════════════════════════════

class ArithmeticCodingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Arithmetic Coding"))
        lay.addWidget(formula_label(
            "Range_new = [low + range·C(x),  low + range·C(x) + range·p(x))\n"
            "Output: any number in final interval"
        ))
        lay.addWidget(info_label(
            "Arithmetic coding represents an entire message as a single number in [0, 1). "
            "It achieves coding efficiency arbitrarily close to entropy by using progressively "
            "smaller sub-intervals."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        dg = QGroupBox("Symbol Distribution")
        dl = QVBoxLayout(dg)
        dl.addWidget(QLabel("Format: symbol:probability (one per line)"))
        self.dist_input = QTextEdit()
        self.dist_input.setMaximumHeight(110)
        self.dist_input.setPlainText("a:0.5\nb:0.3\nc:0.2")
        dl.addWidget(self.dist_input)
        ll.addWidget(dg)

        mg = QGroupBox("Message to Encode")
        ml = QVBoxLayout(mg)
        self.msg_input = QLineEdit("abac")
        ml.addWidget(self.msg_input)
        enc_btn = QPushButton("Encode"); enc_btn.clicked.connect(self._encode)
        ml.addWidget(enc_btn)
        ll.addWidget(mg)

        rg = QGroupBox("Encoding Result")
        rl = QVBoxLayout(rg)
        self.result_label = result_label("Encoded value: —")
        rl.addWidget(self.result_label)
        self.bits_label = QLabel("Bits needed: —")
        self.bits_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#fab387; font-weight:bold;")
        rl.addWidget(self.bits_label)
        self.ac_detail = detail_textedit()
        rl.addWidget(self.ac_detail)
        ll.addWidget(rg)
        content.addWidget(left, 2)

        right = QWidget(); rl2 = QVBoxLayout(right)
        self.canvas = MplCanvas(self, width=6, height=6)
        rl2.addWidget(self.canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._encode()

    def _encode(self):
        try:
            lines = self.dist_input.toPlainText().strip().split('\n')
            dist = {}
            for line in lines:
                if ':' in line:
                    s, p = line.split(':')
                    dist[s.strip()] = float(p.strip())
            if abs(sum(dist.values()) - 1.0) > 0.01:
                self.result_label.setText("⚠  Probabilities must sum to 1.0"); return

            symbols = list(dist.keys())
            probs = [dist[s] for s in symbols]
            cdf = [0.0]
            for p in probs:
                cdf.append(cdf[-1] + p)

            message = self.msg_input.text().strip()
            if not message:
                self.result_label.setText("⚠  Enter a message"); return
            if not all(c in dist for c in message):
                self.result_label.setText(f"⚠  Unknown symbol. Use: {', '.join(symbols)}"); return

            low, high = 0.0, 1.0
            history = [(0, low, high, "Start")]
            detail_lines = [f"Message: {message}\n"]
            detail_lines.append(f"Symbols: {dict(zip(symbols, probs))}\n\n")

            for step, ch in enumerate(message):
                idx = symbols.index(ch)
                prev_low, prev_high = low, high
                rng = high - low
                low = prev_low + rng * cdf[idx]
                high = prev_low + rng * cdf[idx + 1]
                history.append((step + 1, low, high, ch))
                detail_lines.append(
                    f"Step {step+1}: '{ch}'  range=[{prev_low:.8f}, {prev_high:.8f})\n"
                    f"  new=[{low:.8f}, {high:.8f})\n\n"
                )

            encoded = (low + high) / 2
            interval_width = high - low
            bits_needed = max(1, math.ceil(-math.log2(interval_width))) if interval_width > 0 else 1
            h = shannon_entropy(probs)
            h_total = h * len(message)

            self.result_label.setText(f"Encoded value: {encoded:.10f}")
            eff_str = f"  |  Efficiency: {h_total/bits_needed*100:.1f}%" if bits_needed > 0 and h_total > 0 else ""
            self.bits_label.setText(f"Bits needed: {bits_needed}  |  H(msg) = {h_total:.2f}{eff_str}")

            detail_lines.append(f"{'─'*36}\nFinal interval: [{low:.10f}, {high:.10f})\n")
            detail_lines.append(f"Width: {interval_width:.10f}\n")
            detail_lines.append(f"Encoded: {encoded:.10f}\n")
            detail_lines.append(f"Bits: {bits_needed}\n")
            detail_lines.append(f"H(msg): {h_total:.2f} bits\n")
            self.ac_detail.setText(''.join(detail_lines))

            # Draw intervals
            ax = self.canvas.axes; ax.clear()
            n_steps = len(history)
            colors = ['#89b4fa', '#fab387', '#a6e3a1', '#f38ba8', '#cba6f7',
                      '#94e2d5', '#f9e2af', '#74c7ec', '#eba0ac', '#89dcec']

            for i, (step, lo, hi, ch) in enumerate(history):
                y = n_steps - i
                color = colors[step % len(colors)] if step > 0 else '#585b70'
                lw = 3 if step > 0 else 1.5
                ax.plot([lo, hi], [y, y], color=color, linewidth=lw, solid_capstyle='round')
                ax.plot([lo, hi], [y, y], 'o', color=color, markersize=5)
                label = f"Step {step}" if step > 0 else "Initial"
                if ch != "Start": label += f" ('{ch}')"
                ax.text(-0.05, y, label, ha='right', va='center', fontsize=9, color='#a6adc8')
                if hi - lo > 0.01:
                    ax.text((lo + hi) / 2, y + 0.2, f'[{lo:.4f}, {hi:.4f})',
                            ha='center', fontsize=7, color='#7f849c')
                else:
                    ax.text(hi + 0.02, y, f'[{lo:.6f},\n{hi:.6f})',
                            ha='left', va='center', fontsize=7, color='#7f849c')

                if i < 3:
                    for j, s in enumerate(symbols):
                        sub_lo = lo + (hi - lo) * cdf[j]
                        sub_hi = lo + (hi - lo) * cdf[j + 1]
                        ax.plot([sub_lo, sub_hi], [y - 0.15, y - 0.15],
                                color=colors[j % len(colors)], linewidth=1, alpha=0.5)
                        if sub_hi - sub_lo > 0.03:
                            ax.text((sub_lo + sub_hi) / 2, y - 0.35, s, ha='center',
                                    fontsize=7, color=colors[j % len(colors)], alpha=0.7)

            ax.set_xlim(-0.15, 1.05); ax.set_ylim(0, n_steps + 1)
            ax.set_xlabel('Interval [0, 1)')
            ax.set_title('Arithmetic Coding — Interval Narrowing')
            ax.set_yticks([])
            ax.grid(True, alpha=0.1, color='#3a3a55', axis='x')
            self.canvas.fig.tight_layout(); self.canvas.draw()
        except Exception as e:
            self.result_label.setText(f"⚠  Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 14 — MARKOV CHAIN ENTROPY RATE
# ═══════════════════════════════════════════════════════════════════════════════

class MarkovEntropyPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(title_label("Markov Chain Entropy Rate"))
        lay.addWidget(formula_label(
            "H(𝒳) = −Σᵢ μᵢ Σⱼ Pᵢⱼ · log₂ Pᵢⱼ   [bits/symbol]\n"
            "where μ = stationary distribution"
        ))
        lay.addWidget(info_label(
            "For a stationary Markov chain, the entropy rate measures the average "
            "information per symbol. It accounts for correlation between successive symbols."
        ))

        content = QHBoxLayout()
        left = QWidget(); ll = QVBoxLayout(left)

        pg = QGroupBox("Preset Markov Chains")
        pl = QVBoxLayout(pg)
        presets_layout = QHBoxLayout()
        presets = [
            ("Fair Coin", [[0.5, 0.5], [0.5, 0.5]]),
            ("Biased", [[0.9, 0.1], [0.1, 0.9]]),
            ("Mild", [[0.7, 0.3], [0.2, 0.8]]),
            ("Strong", [[0.95, 0.05], [0.05, 0.95]]),
            ("Alternating", [[0.1, 0.9], [0.9, 0.1]]),
        ]
        for name, matrix in presets:
            b = QPushButton(name); b.setObjectName("secondary")
            b.clicked.connect(lambda _, m=matrix: self._set_matrix(m))
            presets_layout.addWidget(b)
        pl.addLayout(presets_layout)
        ll.addWidget(pg)

        mg = QGroupBox("Transition Matrix P")
        ml = QVBoxLayout(mg)
        self.matrix_table = QTableWidget(2, 2)
        self.matrix_table.setHorizontalHeaderLabels(["→ S₀", "→ S₁"])
        self.matrix_table.setVerticalHeaderLabels(["S₀", "S₁"])
        default = [[0.7, 0.3], [0.2, 0.8]]
        for i in range(2):
            for j in range(2):
                it = QTableWidgetItem(str(default[i][j]))
                it.setTextAlignment(Qt.AlignCenter)
                self.matrix_table.setItem(i, j, it)
        self.matrix_table.setMaximumHeight(120)
        self.matrix_table.cellChanged.connect(self._update)
        ml.addWidget(self.matrix_table)
        calc_btn = QPushButton("Calculate"); calc_btn.clicked.connect(self._update)
        ml.addWidget(calc_btn)
        ll.addWidget(mg)

        rg = QGroupBox("Results")
        rl = QVBoxLayout(rg)
        self.mu_label = QLabel("Stationary dist: —")
        self.mu_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#89b4fa;")
        rl.addWidget(self.mu_label)
        self.h_rate_label = result_label("H(𝒳) = — bits/symbol")
        rl.addWidget(self.h_rate_label)
        self.h_indep_label = QLabel("If independent: —")
        self.h_indep_label.setStyleSheet("font-family:'Consolas',monospace; font-size:13px; color:#7f849c;")
        rl.addWidget(self.h_indep_label)
        self.markov_detail = detail_textedit()
        rl.addWidget(self.markov_detail)
        ll.addWidget(rg)
        content.addWidget(left, 2)

        right = QWidget(); rl2 = QVBoxLayout(right)
        self.canvas = MplCanvas(self, width=6, height=6)
        rl2.addWidget(self.canvas)
        content.addWidget(right, 3)

        lay.addLayout(content); lay.addStretch()
        self._update()

    def _set_matrix(self, matrix):
        for i in range(2):
            for j in range(2):
                self.matrix_table.item(i, j).setText(str(matrix[i][j]))
        self._update()

    def _update(self):
        try:
            P = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    P[i][j] = float(self.matrix_table.item(i, j).text())
            for i in range(2):
                if abs(P[i].sum() - 1.0) > 0.01:
                    self.h_rate_label.setText(f"⚠  Row {i} doesn't sum to 1"); return
            if np.any(P < 0):
                self.h_rate_label.setText("⚠  Probabilities must be ≥ 0"); return

            denom = P[0, 1] + P[1, 0]
            if denom < 1e-12:
                mu = np.array([0.5, 0.5])
            else:
                mu = np.array([P[1, 0] / denom, P[0, 1] / denom])

            h_rate = 0.0
            for i in range(2):
                for j in range(2):
                    if P[i, j] > 0 and mu[i] > 0:
                        h_rate -= mu[i] * P[i, j] * math.log2(P[i, j])
            h_indep = shannon_entropy(mu)

            self.mu_label.setText(f"μ = [{mu[0]:.4f}, {mu[1]:.4f}]")
            self.h_rate_label.setText(f"H(𝒳) = {h_rate:.4f} bits/symbol")
            self.h_indep_label.setText(f"If independent: H = {h_indep:.4f} bits/symbol")

            txt = [f"{'═'*38}\n  Markov Chain Entropy Rate\n{'═'*38}\n\n"]
            txt.append(f"P = | {P[0,0]:.4f}  {P[0,1]:.4f} |\n")
            txt.append(f"    | {P[1,0]:.4f}  {P[1,1]:.4f} |\n\n")
            txt.append(f"μ₀ = {mu[0]:.4f},  μ₁ = {mu[1]:.4f}\n\n")
            txt.append(f"{'─'*38}\n  Calculation\n{'─'*38}\n")
            for i in range(2):
                for j in range(2):
                    if P[i, j] > 0:
                        term = -mu[i] * P[i, j] * math.log2(P[i, j])
                        txt.append(f"  −μ{i}·P{i}{j}·log₂(P{i}{j}) = −{mu[i]:.4f}×{P[i,j]:.4f}×{math.log2(P[i,j]):.4f} = {term:.4f}\n")
            txt.append(f"\n  H(𝒳) = {h_rate:.4f} bits/symbol\n")
            txt.append(f"  H(X)  = {h_indep:.4f} bits (independent)\n")
            if h_rate < h_indep - 0.001:
                txt.append(f"\n  ✓ Memory reduces uncertainty by {h_indep - h_rate:.4f} bits\n")
            elif abs(h_rate - h_indep) < 0.001:
                txt.append(f"\n  ≈ No memory effect\n")
            self.markov_detail.setText(''.join(txt))

            # Draw state diagram
            ax = self.canvas.axes; ax.clear(); ax.axis('off')
            ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
            positions = [(-1.5, 0), (1.5, 0)]
            state_colors = ['#89b4fa', '#fab387']
            for i, (x, y) in enumerate(positions):
                circle = MplCircle((x, y), 0.6, facecolor='#151525', edgecolor=state_colors[i], linewidth=2.5)
                ax.add_patch(circle)
                ax.text(x, y + 0.1, f'S{i}', ha='center', va='center', fontsize=16, fontweight='bold', color=state_colors[i])
                ax.text(x, y - 0.25, f'μ={mu[i]:.3f}', ha='center', va='center', fontsize=8, color='#7f849c')
            # Self-loops
            for i, (x, y) in enumerate(positions):
                if P[i, i] > 0.01:
                    sign = 1 if i == 0 else -1
                    theta = np.linspace(0.3 * np.pi, 0.7 * np.pi, 50)
                    lx = x + sign * 0.9 * np.cos(theta)
                    ly = y + 1.2 + 0.5 * np.sin(theta)
                    ax.plot(lx, ly, color=state_colors[i], linewidth=1.5, alpha=0.7)
                    ax.annotate('', xy=(lx[-1], ly[-1]), xytext=(lx[-3], ly[-3]),
                                arrowprops=dict(arrowstyle='->', color=state_colors[i], lw=1.5))
                    ax.text(x + sign * 0.3, y + 1.6, f'P={P[i,i]:.2f}', ha='center', fontsize=9, color=state_colors[i])
            # Cross transitions
            for i, j in [(0, 1), (1, 0)]:
                if P[i, j] > 0.01:
                    x1, y1 = positions[i]; x2, y2 = positions[j]
                    dy = -0.3 if i == 0 else 0.3
                    ax.annotate('', xy=(x2 - 0.65 * (1 if x2 > x1 else -1), y2 + dy),
                                xytext=(x1 + 0.65 * (1 if x2 > x1 else -1), y1 + dy),
                                arrowprops=dict(arrowstyle='->', color='#a6e3a1', lw=2,
                                                connectionstyle=f'arc3,rad={0.2 if i==0 else -0.2}'))
                    ax.text((x1+x2)/2, dy - 0.3, f'P={P[i,j]:.2f}', ha='center', fontsize=9, color='#a6e3a1', fontweight='bold')

            ax.text(0, 2.2, f'Markov Chain   H(𝒳) = {h_rate:.4f} bits/symbol',
                    ha='center', fontsize=13, fontweight='bold', color='#89b4fa')
            self.canvas.fig.tight_layout(); self.canvas.draw()
        except (ValueError, AttributeError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 15 — FORMULA REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class FormulaReferencePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setSpacing(12); lay.setContentsMargins(30, 20, 30, 20)
        lay.addWidget(title_label("Formula Reference"))
        lay.addWidget(info_label(
            "A comprehensive cheat sheet of all key information theory formulas. "
            "Use this as a quick reference while studying."
        ))
        self.ref_text = detail_textedit()
        self.ref_text.setStyleSheet(
            self.ref_text.styleSheet().replace("font-size: 11px", "font-size: 12px"))
        self.ref_text.setText(self._build_reference())
        lay.addWidget(self.ref_text)

    def _build_reference(self):
        return """
═══════════════════════════════════════════════════════════════════
  INFORMATION THEORY — FORMULA REFERENCE
═══════════════════════════════════════════════════════════════════

─────────────────────────────────────
  1. SHANNON ENTROPY
─────────────────────────────────────

  H(X) = −Σ_x  p(x) · log₂ p(x)

  • Measures average uncertainty in X
  • Unit: bits (when log base 2)
  • H(X) ≥ 0, with equality iff X deterministic
  • Maximum: H(X) = log₂|X| when X is uniform

  Binary entropy function:
  H_b(p) = −p·log₂(p) − (1−p)·log₂(1−p)
  Maximum: H_b(0.5) = 1 bit

─────────────────────────────────────
  2. JOINT ENTROPY
─────────────────────────────────────

  H(X,Y) = −Σ_{x,y}  p(x,y) · log₂ p(x,y)

  • H(X,Y) ≤ H(X) + H(Y)  (subadditivity)
  • Equality iff X, Y independent

─────────────────────────────────────
  3. CONDITIONAL ENTROPY
─────────────────────────────────────

  H(Y|X) = H(X,Y) − H(X)

  • H(Y|X) ≤ H(Y)  (conditioning reduces entropy)
  • Chain rule: H(X₁,...,Xₙ) = Σᵢ H(Xᵢ|X₁,...,Xᵢ₋₁)

─────────────────────────────────────
  4. MUTUAL INFORMATION
─────────────────────────────────────

  I(X;Y) = H(X) − H(X|Y) = H(Y) − H(Y|X)
         = H(X) + H(Y) − H(X,Y)

  • I(X;Y) ≥ 0,  I(X;Y) = 0  iff  X ⊥ Y
  • Symmetric: I(X;Y) = I(Y;X)

  Conditional: I(X;Y|Z) = H(X|Z) − H(X|Y,Z)

─────────────────────────────────────
  5. KL DIVERGENCE
─────────────────────────────────────

  D_KL(P‖Q) = Σ_x  P(x) · log₂(P(x)/Q(x))

  • D_KL(P‖Q) ≥ 0  (Gibbs' inequality)
  • NOT symmetric, NOT a metric

  Cross entropy:  H(P,Q) = H(P) + D_KL(P‖Q)

─────────────────────────────────────
  6. SOURCE CODING THEOREM
─────────────────────────────────────

  H(X) ≤ L̄ < H(X) + 1    (symbol codes)

  Kraft inequality:  Σᵢ 2^{−lᵢ} ≤ 1

─────────────────────────────────────
  7. CHANNEL CAPACITY
─────────────────────────────────────

  C = max_{p(x)}  I(X;Y)

  BSC(p):  C = 1 − H_b(p)
  BEC(ε):  C = 1 − ε

─────────────────────────────────────
  8. NOISY CHANNEL CODING THEOREM
─────────────────────────────────────

  R < C:  ∃ code with P_e → 0 as n → ∞
  R > C:  P_e bounded away from 0

  P_e ≤ 2^{−n·E(R)}

─────────────────────────────────────
  9. FANO'S INEQUALITY
─────────────────────────────────────

  H(P_e) + P_e · log₂(|X̂|−1) ≥ H(X|Y)

  P_e ≥ (H(X|Y) − 1) / log₂(|X̂|)

─────────────────────────────────────
  10. DATA PROCESSING INEQUALITY
─────────────────────────────────────

  X → Y → Z  ⟹  I(X;Z) ≤ I(X;Y)

─────────────────────────────────────
  11. MARKOV CHAIN ENTROPY RATE
─────────────────────────────────────

  H(𝒳) = −Σᵢ  μᵢ Σⱼ  Pᵢⱼ · log₂ Pᵢⱼ

  H(𝒳) ≤ H(X)  (memory reduces rate)

─────────────────────────────────────
  12. HAMMING(7,4)
─────────────────────────────────────

  4 data + 3 parity bits, min distance = 3
  Corrects 1 error, detects 2
  Rate: 4/7 ≈ 0.571

═══════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class InfoTheoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Information Theory — Interactive Learning")
        self.setMinimumSize(1280, 800)
        self.resize(1500, 940)
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Sidebar ──
        sidebar = QWidget()
        sidebar.setFixedWidth(255)
        sidebar.setStyleSheet("background-color: #0c0c18; border-right: 1px solid #1e1e35;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(3)

        logo = QLabel("📐 InfoTheory")
        logo.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #89b4fa; "
            "padding: 10px; background: transparent;"
        )
        sidebar_layout.addWidget(logo)
        sub = QLabel("Interactive Learning App")
        sub.setStyleSheet("font-size: 11px; color: #585b70; padding: 0 10px 8px; background: transparent;")
        sidebar_layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2a2a40; max-height: 1px; margin: 5px 5px 8px;")
        sidebar_layout.addWidget(sep)

        self.nav_list = QListWidget()
        self.nav_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; }
            QListWidget::item { padding: 10px 14px; border-radius: 8px; margin: 1px 4px;
                                color: #7f849c; font-size: 13px; }
            QListWidget::item:selected { background-color: #2d2d50; color: #89b4fa; font-weight: bold; }
            QListWidget::item:hover  { background-color: #1e1e35; color: #a6adc8; }
        """)

        topics = [
            "🏠  Dashboard",
            "📐  Shannon Entropy",
            "🔗  Joint & Conditional",
            "🔄  Mutual Information",
            "🌲  Huffman Coding",
            "📡  Channel Capacity",
            "📏  KL Divergence",
            "💡  Source Coding Theorem",
            "🧪  Practice Quiz",
            "📡  Noisy Channel Theorem",
            "📐  Fano's Inequality",
            "🛡️  Hamming Code",
            "🔢  Arithmetic Coding",
            "⛓️  Markov Entropy Rate",
            "📖  Formula Reference",
        ]
        self.nav_list.addItems(topics)
        self.nav_list.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.nav_list)
        sidebar_layout.addStretch()

        footer = QLabel("PySide6 · Matplotlib · NumPy")
        footer.setStyleSheet("color: #45475a; font-size: 10px; padding: 5px;")
        footer.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(footer)
        main_layout.addWidget(sidebar)

        # ── Page stack ──
        self.stack = QStackedWidget()
        self.pages = [
            DashboardPage(),
            EntropyPage(),
            JointEntropyPage(),
            MutualInfoPage(),
            HuffmanPage(),
            ChannelCapacityPage(),
            KLDivergencePage(),
            SourceCodingPage(),
            QuizPage(),
            NoisyChannelPage(),
            FanoPage(),
            HammingCodePage(),
            ArithmeticCodingPage(),
            MarkovEntropyPage(),
            FormulaReferencePage(),
        ]
        for page in self.pages:
            scroll = QScrollArea()
            scroll.setWidget(page)
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
            self.stack.addWidget(scroll)

        main_layout.addWidget(self.stack, 1)
        self.nav_list.setCurrentRow(0)

    def _switch_page(self, index):
        self.stack.setCurrentIndex(index)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor("#0f0f1a"))
    palette.setColor(palette.ColorRole.WindowText, QColor("#cdd6f4"))
    palette.setColor(palette.ColorRole.Base, QColor("#151525"))
    palette.setColor(palette.ColorRole.Text, QColor("#cdd6f4"))
    palette.setColor(palette.ColorRole.Button, QColor("#2a2a40"))
    palette.setColor(palette.ColorRole.ButtonText, QColor("#cdd6f4"))
    palette.setColor(palette.ColorRole.Highlight, QColor("#89b4fa"))
    palette.setColor(palette.ColorRole.HighlightedText, QColor("#0f0f1a"))
    app.setPalette(palette)

    window = InfoTheoryApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()