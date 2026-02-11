"""
Embedded GUI for the agent workflow.
"""
import asyncio
import csv
import datetime
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QListWidget, QFrame, QFileDialog,
    QListWidgetItem, QScrollArea, QSizePolicy, QStackedWidget, QMessageBox, QMenu, QGroupBox, QComboBox, QFormLayout,
    QToolTip, QSpinBox, QDoubleSpinBox, QProgressBar,
    QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QSize, QTimer, QEvent, QThread, QUrl
from PyQt6.QtGui import QFont, QIcon, QPixmap, QTextCursor, QTextDocument, QTextOption, QImage, QPainter, QDesktopServices
from PyQt6.QtCore import pyqtSignal, QObject
from utils import get_swi_thumbnail


# Support running directly: python agentor/gui3.py
# Ensure repo-root packages can be imported (e.g., `langchain_openai/`, `pathomics/`, `example/`).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wsi_agent import WSIAgent, ToolCallbackHandler


class AgentWorker(QObject):

    finished = pyqtSignal()

    def __init__(self, agent_executor, question, signals):
        super().__init__()
        self.agent_executor = agent_executor
        self.question = question
        self.signals = signals
        self._is_running = True

    def run(self):
        try:
            # Run in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Execute agent
            result = self.agent_executor.run_agent(self.question)
            print(f"AgentWorker run result :{result}")
            self.signals.tool_end.emit(result)

        except Exception as e:
            self.signals.error_occurred.emit(f"Execution error: {str(e)}")
        finally:
            self.finished.emit()
            if loop and not loop.is_closed():
                loop.close()

    def stop(self):
        self._is_running = False


class ToolSignals(QObject):
    tool_end = pyqtSignal(str)         # tool output
    error_occurred = pyqtSignal(str)   # error message (e.g., missing API key)


class AppendOutputEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, text):
        super().__init__(self.EVENT_TYPE)
        self.text = text


"""
Note: A custom "segmentation results preview" dialog used to exist here, but the
expected UX is consistent with selecting a WSI folder:
use the system "Select Folder" dialog only; after selection, print subfolder names
in the chat area.
"""


class ChatBubble(QWidget):
    """Chat bubble widget."""

    def __init__(self, text, is_ai=False, parent=None):
        super().__init__(parent)
        self.is_ai = is_ai
        layout1 = QVBoxLayout()
        # Main layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Avatar
        self.avatar = QLabel()
        self.avatar.setFixedSize(40, 40)
        self.avatar.setScaledContents(True)

        # Avatar by role
        if is_ai:
            ai_pixmap = QPixmap(":ai-icon.png")
            if ai_pixmap.isNull():
                # self.avatar.setText("🤖")
                self.avatar.setText("A")
                self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.avatar.setStyleSheet("""
                    background-color: #e0e0e0;
                    border-radius: 20px;
                    font-size: 20px;
                """)
            else:
                ai_icon = ai_pixmap.scaled(
                    40,
                    40,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.avatar.setPixmap(ai_icon)
        else:
            user_pixmap = QPixmap(":user-icon.png")
            if user_pixmap.isNull():
                # self.avatar.setText("👤")
                self.avatar.setText("U")
                self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.avatar.setStyleSheet("""
                    background-color: #e0e0e0;
                    border-radius: 20px;
                    font-size: 20px;
                """)
            else:
                user_icon = user_pixmap.scaled(
                    40,
                    40,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.avatar.setPixmap(user_icon)

        # Chat text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # self.text_edit.setText(text)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Bubble styles
        if is_ai:
            self.text_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #e3f2fd;
                    border-radius: 15px;
                    padding: 10px;
                    border: 1px solid #bbdefb;
                }
            """)
        else:
            self.text_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #e8f5e9;
                    border-radius: 15px;
                    padding: 10px;
                    border: 1px solid #c8e6c9;
                }
            """)
        # Adjust text edit height
        doc = self.text_edit.document()
        doc.setTextWidth(self.text_edit.width())

        # Adjust text edit height
        doc = self.text_edit.document()

        str_ = re.findall('[a-zA-Z]', text)
        if not text:
            text = "no message"
        per_num = 104 if len(str_) / len(text) > 0.5 else 50
        if is_ai:
            line_num = len(text.split("\n"))
            if line_num == 1 and len(text) > per_num:
                line_num = int(len(text) / per_num) + 1
            else:
                line_num += 1
        else:
            line_num = int(len(text) / per_num) + 1

        doc.setPlainText(text)
        doc.setTextWidth(self.text_edit.width())
        # height = int(doc.size().height())  # include padding
        if line_num == 1:
            height = int(doc.size().height()) + 20
        else:
            height = line_num * 20 + 20

        # height = int(doc.size().height()) + 20  # include padding
        self.text_edit.setFixedHeight(height)
        # Add widgets to layout
        if is_ai:
            layout.addWidget(self.avatar)
            layout.addWidget(self.text_edit, 1)
        else:
            layout.addWidget(self.text_edit, 1)
            layout.addWidget(self.avatar)
        time_label = QLabel()
        time_label.setText(f"{str(datetime.datetime.now())[:19]}")
        time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_label.setStyleSheet("""
                            background-color: #f5f9fc;
                            font-size: 13px;
                        """)
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.addWidget(time_label)
        layout1.addLayout(layout)
        self.setLayout(layout1)

    # def sizeHint(self):
    #     # Calculate a suitable widget size
    #     doc = self.text_edit.document()
    #     doc.setTextWidth(self.text_edit.width())
    #     height = doc.size().height() + 35  # include padding and avatar height
    #     return QSize(self.width(), int(height))

    def getSizeHint(self):
        # Calculate a suitable widget size
        height = self.text_edit.height() + 60
        return QSize(self.width(), int(height))


class PatchPreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        hbox = QHBoxLayout(self)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("background-color: #f0f0f0; border-radius: 8px;")
        bordered_pixmap = QPixmap(r'D:\work_space\backend\TestPost\agent\123.png')
        bordered_pixmap = bordered_pixmap.scaled(
            512, 512,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        image_label.setPixmap(bordered_pixmap)
        hbox.addWidget(image_label)
        self.setWindowTitle('Patch Preview')
        self.show()

    def contextMenuEvent(self, event):

        cmenu = QMenu(self)
        pre_wsi = cmenu.addAction("Previous")
        next_wsi = cmenu.addAction("Next")
        mask_view = cmenu.addAction("Mask")
        action = cmenu.exec(self.mapToGlobal(event.pos()))

        if action == pre_wsi:

            QMessageBox.about(self, 'Info', 'This Is Previous Patch')
        elif action == next_wsi:
            QMessageBox.about(self, 'Info', 'This Is Next Patch')
        elif action == mask_view:
            QMessageBox.about(self, 'Info', 'This Is Mask')


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        QToolTip.setFont(QFont('Times New Roman', 15))  # tooltip font and size
        le = QLineEdit()
        # le.setFixedHeight(24)
        self.setLineEdit(le)
        self.lineEdit().setReadOnly(True)
        self.view().clicked.connect(self.selectItemAction)
        self.addCheckableItem("Select All")
        self.SelectAllStatus = 1

    def addCheckableItem(self, text):
        super().addItem(text)
        item = self.model().item(self.count() - 1, 0)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Unchecked)
        item.setToolTip(text)

    def addCheckableItems(self, texts):
        for text in texts:
            self.addCheckableItem(text)

    def ifChecked(self, index):
        item = self.model().item(index, 0)
        return item.checkState() == Qt.CheckState.Checked

    def checkedItems(self):
        return [self.itemText(i) for i in range(self.count()) if self.ifChecked(i)]

    def checkedItemsStr(self):
        items = [x for x in self.checkedItems() if x != "Select All"]
        return ";".join(items).strip(";")

    def showPopup(self):
        self.view().setMinimumWidth(3 * self.width() // 2)  # widen dropdown list
        self.view().setMaximumHeight(200)  # max dropdown height
        super().showPopup()

    def selectItemAction(self, index):
        if index.row() == 0:
            for i in range(self.model().rowCount()):
                if self.SelectAllStatus:
                    self.model().item(i).setCheckState(Qt.CheckState.Checked)
                else:
                    self.model().item(i).setCheckState(Qt.CheckState.Unchecked)
            self.SelectAllStatus = (self.SelectAllStatus + 1) % 2

        self.lineEdit().clear()
        self.lineEdit().setText(self.checkedItemsStr())

    def clear(self) -> None:
        super().clear()
        self.addCheckableItem("Select All")

    def select_all(self):
        for i in range(self.model().rowCount()):
            self.model().item(i).setCheckState(Qt.CheckState.Checked)
        self.lineEdit().setText(self.checkedItemsStr())

    def select_one(self, i):
        self.model().item(i).setCheckState(Qt.CheckState.Checked)
        self.lineEdit().setText(self.checkedItemsStr())


class ParamsSettingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """"""
        layout = QVBoxLayout()

        seg_box = QGroupBox("1. SegmentImage")
        seg_layout_row = QHBoxLayout()

        seg_layout = QHBoxLayout()
        seg_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.seg_type_widget = QComboBox()
        self.seg_type_widget.setFixedWidth(80)
        self.seg_type_widget.addItems(["all", "tissue", "cell"])
        seg_layout.addWidget(QLabel("SegType:"))
        seg_layout.addWidget(self.seg_type_widget)
        seg_layout_row.addLayout(seg_layout)
        seg_box.setLayout(seg_layout_row)
        ###---------------------------------------
        extract_box = QGroupBox("2. ExtractFeatures")
        extract_form_layout = QFormLayout()
        extract_layout_1 = QHBoxLayout()
        sub_extract_layout_1_1 = QHBoxLayout()
        sub_extract_layout_1_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.extract_type_widget = QComboBox()
        self.extract_type_widget.setFixedWidth(80)
        self.extract_type_widget.addItems(["all", "tissue", "cell"])
        sub_extract_layout_1_1.addWidget(QLabel("ExtractType:"))
        sub_extract_layout_1_1.addWidget(self.extract_type_widget)

        sub_extract_layout_1_2 = QHBoxLayout()
        sub_extract_layout_1_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.suffix_widget = QComboBox()
        self.suffix_widget.addItems(["png"])
        self.suffix_widget.setFixedWidth(80)
        sub_extract_layout_1_2.addWidget(QLabel("MaskSuffix:"))
        sub_extract_layout_1_2.addWidget(self.suffix_widget)

        sub_extract_layout_1_3 = QHBoxLayout()
        sub_extract_layout_1_3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.feature_type_widget = QComboBox()
        self.feature_type_widget.setFixedWidth(80)
        self.feature_type_widget.addItems(["shape", "texture", "topology", "interplay", "all"])
        sub_extract_layout_1_3.addWidget(QLabel("FeatureType:"))
        sub_extract_layout_1_3.addWidget(self.feature_type_widget)

        sub_extract_layout_1_4 = QHBoxLayout()
        sub_extract_layout_1_4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.nuclei_type_widget = QComboBox()
        self.nuclei_type_widget.setFixedWidth(80)
        self.nuclei_type_widget.addItems(['None', '1', '2', '3', '4', '5'])
        sub_extract_layout_1_4.addWidget(QLabel("NucleiType:"))
        sub_extract_layout_1_4.addWidget(self.nuclei_type_widget)
        extract_layout_1.addLayout(sub_extract_layout_1_1, 1)
        extract_layout_1.addLayout(sub_extract_layout_1_2, 1)
        extract_layout_1.addLayout(sub_extract_layout_1_3, 1)
        extract_layout_1.addLayout(sub_extract_layout_1_4, 1)

        extract_layout_2 = QHBoxLayout()
        sub_extract_layout_2_1 = QHBoxLayout()
        sub_extract_layout_2_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.wsi_num_widget = QSpinBox()
        self.wsi_num_widget.setMaximum(10000)
        self.wsi_num_widget.setValue(200)
        self.wsi_num_widget.setFixedWidth(80)
        sub_extract_layout_2_1.addWidget(QLabel("wsi_num:"))
        sub_extract_layout_2_1.addWidget(self.wsi_num_widget)

        sub_extract_layout_2_2 = QHBoxLayout()
        sub_extract_layout_2_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.mag_widget = QSpinBox()
        self.mag_widget.setMaximum(10000)
        self.mag_widget.setValue(40)
        self.mag_widget.setFixedWidth(80)
        sub_extract_layout_2_2.addWidget(QLabel("mag:"))
        sub_extract_layout_2_2.addWidget(self.mag_widget)

        sub_extract_layout_2_3 = QHBoxLayout()
        sub_extract_layout_2_3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.n_work_widget1 = QSpinBox()
        self.n_work_widget1.setValue(8)
        self.n_work_widget1.setFixedWidth(80)
        sub_extract_layout_2_3.addWidget(QLabel('n_work:'))
        sub_extract_layout_2_3.addWidget(self.n_work_widget1)

        extract_layout_2.addLayout(sub_extract_layout_2_1)
        extract_layout_2.addLayout(sub_extract_layout_2_2)
        extract_layout_2.addLayout(sub_extract_layout_2_3)

        extract_form_layout.addRow(extract_layout_1)
        extract_form_layout.addRow(extract_layout_2)
        extract_box.setLayout(extract_form_layout)
        #### -------------- Model construction -----------------
        model_box = QGroupBox("3. ModelConstructor")
        model_form_layout = QFormLayout()
        model_layout_1 = QHBoxLayout()
        sub_model_layout_1_1 = QHBoxLayout()
        sub_model_layout_1_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top_feature_widget = QSpinBox()
        self.top_feature_widget.setMaximum(10000)
        self.top_feature_widget.setValue(6)
        self.top_feature_widget.setFixedWidth(80)
        sub_model_layout_1_1.addWidget(QLabel("top_feature_num:"))
        sub_model_layout_1_1.addWidget(self.top_feature_widget)

        sub_model_layout_1_2 = QHBoxLayout()
        sub_model_layout_1_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.k_widget = QSpinBox()
        self.k_widget.setMaximum(10000)
        self.k_widget.setValue(5)
        self.k_widget.setFixedWidth(80)
        sub_model_layout_1_2.addWidget(QLabel("k_fold:"))
        sub_model_layout_1_2.addWidget(self.k_widget)

        sub_model_layout_1_3 = QHBoxLayout()
        sub_model_layout_1_3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.feature_score_widget = QComboBox()
        self.feature_score_widget.setFixedWidth(80)
        self.feature_score_widget.addItems(["addone", "weighted"])
        sub_model_layout_1_3.addWidget(QLabel("feature_score_method:"))
        sub_model_layout_1_3.addWidget(self.feature_score_widget)

        sub_model_layout_1_4 = QHBoxLayout()
        sub_model_layout_1_4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.thresh_widget = QDoubleSpinBox()
        self.thresh_widget.setMaximum(10000)
        self.thresh_widget.setValue(0)
        self.thresh_widget.setFixedWidth(80)
        sub_model_layout_1_4.addWidget(QLabel("var_thresh:"))
        sub_model_layout_1_4.addWidget(self.thresh_widget)

        model_layout_1.addLayout(sub_model_layout_1_1, 1)
        model_layout_1.addLayout(sub_model_layout_1_2, 1)
        model_layout_1.addLayout(sub_model_layout_1_3, 1)
        model_layout_1.addLayout(sub_model_layout_1_4, 1)

        model_layout_2 = QHBoxLayout()
        sub_model_layout_2_1 = QHBoxLayout()
        sub_model_layout_2_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.corr_widget = QDoubleSpinBox()
        self.corr_widget.setValue(0.9)
        self.corr_widget.setFixedWidth(80)
        sub_model_layout_2_1.addWidget(QLabel("corr_threshold:"))
        sub_model_layout_2_1.addWidget(self.corr_widget)

        sub_model_layout_2_2 = QHBoxLayout()
        sub_model_layout_2_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.repeats_widget = QSpinBox()
        self.repeats_widget.setMaximum(10000)
        self.repeats_widget.setValue(100)
        self.repeats_widget.setFixedWidth(80)
        sub_model_layout_2_2.addWidget(QLabel("repeats_num:"))
        sub_model_layout_2_2.addWidget(self.repeats_widget)

        sub_model_layout_2_3 = QHBoxLayout()
        sub_model_layout_2_3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.n_work_widget2 = QSpinBox()
        # Model Constructor default parallel workers
        self.n_work_widget2.setValue(8)
        self.n_work_widget2.setFixedWidth(80)
        sub_model_layout_2_3.addWidget(QLabel("n_work:"))
        sub_model_layout_2_3.addWidget(self.n_work_widget2)
        model_layout_2.addLayout(sub_model_layout_2_1)
        model_layout_2.addLayout(sub_model_layout_2_2)
        model_layout_2.addLayout(sub_model_layout_2_3)

        model_layout_3 = QHBoxLayout()

        self.feats_combobox = CheckableComboBox(self)
        self.feats_combobox.setFixedHeight(30)
        self.feats_combobox.addCheckableItems(['Lasso', 'XGBoost', 'RandomForest', 'Elastic-Net', 'RFE', 'Univariate', 'mrmr', 'ttest', 'ranksums', 'mutualInfo'])
        # Default selection: ranksums + mutualInfo (aligned with Btrain_test_cross_validationV3.py examples)
        self.feats_combobox.select_one(9)   # ranksums (based on current item order, 1-based)
        self.feats_combobox.select_one(10)  # mutualInfo
        t = QLabel("feats_selection:")
        t.setFixedWidth(100)
        model_layout_3.addWidget(t, 1)
        model_layout_3.addWidget(self.feats_combobox, 10)

        model_layout_4 = QHBoxLayout()
        self.classifers_combobox = CheckableComboBox(self)
        self.classifers_combobox.setFixedHeight(30)
        self.classifers_combobox.addCheckableItems(
            ['QDA', 'LDA', 'RandomForest', 'DecisionTree', 'KNeigh', 'LinearSVC', 'MLP', 'GaussianNB', 'SGD', 'SVC_rbf', 'AdaBoost'])
        # Default selection: LinearSVC + LDA + RandomForest
        self.classifers_combobox.select_one(6)  # LinearSVC
        self.classifers_combobox.select_one(2)  # LDA
        self.classifers_combobox.select_one(3)  # RandomForest
        t1 = QLabel("classifers:")
        t1.setFixedWidth(100)
        model_layout_4.addWidget(t1, 1)
        model_layout_4.addWidget(self.classifers_combobox, 10)

        model_form_layout.addRow(model_layout_1)
        model_form_layout.addRow(model_layout_2)
        model_form_layout.addRow(model_layout_3)
        model_form_layout.addRow(model_layout_4)
        model_box.setLayout(model_form_layout)
        ### -------------- Genomics analysis -------------------
        gene_box = QGroupBox("4. GenomicsAnalyzer")
        gene_form_layout = QFormLayout()
        gene_layout_1 = QHBoxLayout()
        gene_layout_1_1 = QHBoxLayout()
        gene_layout_1_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top_widget2 = QSpinBox()
        self.top_widget2.setValue(6)
        self.top_widget2.setFixedWidth(80)
        gene_layout_1_1.addWidget(QLabel("top:"))
        gene_layout_1_1.addWidget(self.top_widget2)

        gene_layout_1_2 = QHBoxLayout()
        gene_layout_1_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.signif_widget = QSpinBox()
        self.signif_widget.setMaximum(10000)
        self.signif_widget.setValue(2000)
        self.signif_widget.setFixedWidth(80)
        gene_layout_1_2.addWidget(QLabel("top_signif:"))
        gene_layout_1_2.addWidget(self.signif_widget)

        gene_layout_1_3 = QHBoxLayout()
        gene_layout_1_3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.source_widget = QComboBox()
        self.source_widget.setFixedWidth(80)
        self.source_widget.addItems(["GO:All", "GO:MF", "GO:CC", "GO:BP", "KEGG", "REAC", "WP", "TF", "MIRNA", "HPA", "CORUM", "HP"])
        gene_layout_1_3.addWidget(QLabel("source:"))
        gene_layout_1_3.addWidget(self.source_widget)

        gene_layout_1_4 = QHBoxLayout()
        gene_layout_1_4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.category_widget = QComboBox()
        self.category_widget.setFixedWidth(80)
        self.category_widget.addItems(["True", "False"])
        gene_layout_1_4.addWidget(QLabel("category:"))
        gene_layout_1_4.addWidget(self.category_widget)

        gene_layout_1.addLayout(gene_layout_1_1)
        gene_layout_1.addLayout(gene_layout_1_2)
        gene_layout_1.addLayout(gene_layout_1_3)
        gene_layout_1.addLayout(gene_layout_1_4)

        gene_layout_2 = QHBoxLayout()
        gene_layout_2_1 = QHBoxLayout()
        gene_layout_2_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # clinical_col: editable dropdown; supports auto-loading Clinical CSV headers
        self.clinical_widget = QComboBox()
        self.clinical_widget.setEditable(True)
        self.clinical_widget.setFixedHeight(30)
        # Provide a common default to avoid blocking first-time users
        self.clinical_widget.addItems(["T_stage"])
        if self.clinical_widget.lineEdit():
            self.clinical_widget.lineEdit().setPlaceholderText("e.g., T_stage")
        t3 = QLabel("clinical_col:")
        t3.setFixedWidth(100)
        gene_layout_2_1.addWidget(t3)
        gene_layout_2_1.addWidget(self.clinical_widget)
        gene_layout_2.addLayout(gene_layout_2_1)

        gene_form_layout.addRow(gene_layout_1)
        gene_form_layout.addRow(gene_layout_2)
        gene_box.setLayout(gene_form_layout)

        ### -------------- LLM Configuration -------------------
        llm_box = QGroupBox("5. LLM Configuration")
        llm_form_layout = QFormLayout()
        
        # Row 1: API Key and Base URL
        llm_layout_1 = QHBoxLayout()
        llm_layout_1_1 = QHBoxLayout()
        llm_layout_1_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.api_key_widget = QLineEdit()
        self.api_key_widget.setFixedWidth(250)
        self.api_key_widget.setEchoMode(QLineEdit.EchoMode.Password)  # Hide API key
        self.api_key_widget.setPlaceholderText("sk-xxxxx or leave empty to use env/secrets")
        # Try to load from secrets.local.json or environment
        try:
            import sys
            sys.path.insert(0, str(_REPO_ROOT / "agentor"))
            from wsi_agent import _get_env
            default_key = _get_env("OPENAI_API_KEY", "") or _get_env("DASHSCOPE_API_KEY", "")
            if default_key:
                self.api_key_widget.setText(default_key)
        except Exception:
            pass
        llm_layout_1_1.addWidget(QLabel("API Key:"))
        llm_layout_1_1.addWidget(self.api_key_widget)
        
        llm_layout_1_2 = QHBoxLayout()
        llm_layout_1_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.base_url_widget = QLineEdit()
        self.base_url_widget.setFixedWidth(350)
        self.base_url_widget.setPlaceholderText("e.g., https://dashscope.aliyuncs.com/compatible-mode/v1")
        # Load default from secrets/env
        try:
            default_url = _get_env("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.base_url_widget.setText(default_url)
        except Exception:
            self.base_url_widget.setText("https://dashscope.aliyuncs.com/compatible-mode/v1")
        llm_layout_1_2.addWidget(QLabel("Base URL:"))
        llm_layout_1_2.addWidget(self.base_url_widget)
        
        llm_layout_1.addLayout(llm_layout_1_1)
        llm_layout_1.addLayout(llm_layout_1_2)
        
        # Row 2: Agent Model and Report Model
        llm_layout_2 = QHBoxLayout()
        llm_layout_2_1 = QHBoxLayout()
        llm_layout_2_1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.agent_model_widget = QComboBox()
        self.agent_model_widget.setEditable(True)
        self.agent_model_widget.setFixedWidth(200)
        self.agent_model_widget.addItems([
            "qwen3-max",
            "deepseek-r1",
            "qwen-turbo",
            "qwen-plus",
            "gpt-4",
            "gpt-3.5-turbo"
        ])
        # Load default from secrets/env
        try:
            default_model = _get_env("OPENAI_MODEL", "qwen3-max")
            self.agent_model_widget.setCurrentText(default_model)
        except Exception:
            self.agent_model_widget.setCurrentText("qwen3-max")
        llm_layout_2_1.addWidget(QLabel("Agent Model:"))
        llm_layout_2_1.addWidget(self.agent_model_widget)
        
        llm_layout_2_2 = QHBoxLayout()
        llm_layout_2_2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.report_model_widget = QComboBox()
        self.report_model_widget.setEditable(True)
        self.report_model_widget.setFixedWidth(200)
        self.report_model_widget.addItems([
            "qwen-long",
            "qwen3-max",
            "qwen-plus",
            "deepseek-r1",
            "gpt-4"
        ])
        self.report_model_widget.setCurrentText("qwen-long")
        llm_layout_2_2.addWidget(QLabel("Report Model:"))
        llm_layout_2_2.addWidget(self.report_model_widget)
        
        llm_layout_2.addLayout(llm_layout_2_1)
        llm_layout_2.addLayout(llm_layout_2_2)
        
        llm_form_layout.addRow(llm_layout_1)
        llm_form_layout.addRow(llm_layout_2)
        llm_box.setLayout(llm_form_layout)

        layout.addWidget(seg_box)
        layout.addWidget(extract_box)
        layout.addWidget(model_box)
        layout.addWidget(gene_box)
        layout.addWidget(llm_box)
        self.setLayout(layout)
        self.setWindowTitle('Optimized Prototype Interface')

    def get_val(self):
        """"""
        res = {
            "segment_image": {"seg_type": self.seg_type_widget.currentText()},
            "extract_features": {
                "extract_type": self.extract_type_widget.currentText(),
                "mask_suffix": self.suffix_widget.currentText(),
                "feature_type": self.feature_type_widget.currentText(),
                "nuclei_type": int(self.nuclei_type_widget.currentText()) if self.nuclei_type_widget.currentText() != "None" else None,
                "wsi_num": self.wsi_num_widget.value(),
                "mag": self.mag_widget.value(),
                "n_workers": self.n_work_widget1.value()
            },
            "build_model": {
                "top_feature_num": self.top_feature_widget.value(),
                "k_fold": self.k_widget.value(),
                "feature_score_method": self.feature_score_widget.currentText(),
                "var_thresh": self.thresh_widget.value(),
                "corr_threshold": self.corr_widget.value(),
                "repeats_num": self.repeats_widget.value(),
                "n_workers": self.n_work_widget2.value(),
                "list_feats_selection_args": self.feats_combobox.lineEdit().text().split(";") if self.feats_combobox.lineEdit() else [],
                "list_classifers_args": self.classifers_combobox.lineEdit().text().split(";") if self.classifers_combobox.lineEdit() else []
            },
            "analyze_results": {
                "top": self.top_widget2.value(),
                "top_signif": self.signif_widget.value(),
                "sources": self.source_widget.currentText(),
                "category": self.category_widget.currentText() == "True",
                "clinical_col": self.clinical_widget.currentText().strip()
            },
            "llm_config": {
                "api_key": self.api_key_widget.text().strip() if self.api_key_widget.text().strip() else None,
                "base_url": self.base_url_widget.text().strip() if self.base_url_widget.text().strip() else "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "agent_model": self.agent_model_widget.currentText(),
                "report_model": self.report_model_widget.currentText()
            }
        }
        return res

    def update_clinical_columns(self, columns):
        """
        Refresh the dropdown with Clinical CSV header columns and default to T_stage if present.
        columns: list[str]
        """
        try:
            cols = [str(c).strip() for c in (columns or []) if str(c).strip()]
        except Exception:
            cols = []

        # Keep at least one default value to avoid downstream validation failures
        if not cols:
            cols = ["T_stage"]

        current = self.clinical_widget.currentText().strip()
        self.clinical_widget.clear()
        self.clinical_widget.addItems(cols)

        # Prefer T_stage; otherwise keep user's existing input; otherwise pick the first column
        preferred = "T_stage"
        if preferred in cols:
            self.clinical_widget.setCurrentText(preferred)
        elif current:
            # Allow a manually typed column not present in header (some CSVs may be renamed/prefixed)
            self.clinical_widget.setCurrentText(current)
        else:
            self.clinical_widget.setCurrentIndex(0)


class PreviewWidget(QWidget):
    """WSI preview widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_main = parent
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(300)

        # Stacked layout to switch preview states
        self.stacked_widget = QStackedWidget()

        # Default state - no image
        self.empty_state = QLabel("No WSI Image Preview Available")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setStyleSheet("""
            QLabel {
                color: #888888;
                font-style: italic;
                font-size: 14px;
                background-color: #f9f9f9;
                border: 2px dashed #cccccc;
                border-radius: 10px;
            }
        """)

        # Image preview state
        self.image_container = QWidget()
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border-radius: 8px;")

        # Bottom info bar
        info_layout = QHBoxLayout()
        self.image_name = QLabel("File: Not selected")
        self.image_name.setStyleSheet("color: #555555; font-size: 12px;")

        self.image_size = QLabel("Dimensions: N/A")
        self.image_size.setStyleSheet("color: #555555; font-size: 12px;")

        info_layout.addWidget(self.image_name, 7)
        info_layout.addWidget(self.image_size, 3)

        image_layout.addWidget(self.image_label, 8)
        image_layout.addLayout(info_layout, 1)
        self.image_container.setLayout(image_layout)

        # Loading state
        self.loading_state = QLabel("Loading WSI Preview...")
        self.loading_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_state.setStyleSheet("""
            QLabel {
                color: #555555;
                font-size: 14px;
                background-color: #f9f9f9;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
            }
        """)

        # Add states to the stack
        self.stacked_widget.addWidget(self.empty_state)
        self.stacked_widget.addWidget(self.image_container)
        self.stacked_widget.addWidget(self.loading_state)
        self.stacked_widget.setCurrentIndex(0)  # show empty state by default

        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("WSI Preview")
        title_font = QFont("Arial", 12, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #2c3e50;")

        layout.addWidget(title)
        layout.addWidget(self.stacked_widget, 1)

        self.setLayout(layout)

    def set_preview_image(self, image_path):
        """Set the preview image."""
        # Show loading state
        self.stacked_widget.setCurrentIndex(2)
        QApplication.processEvents()  # force UI update

        # Simulate a short loading delay
        QTimer.singleShot(500, lambda: self._load_image(image_path))

    def _load_image(self, image_path):
        """Load and display the image."""
        try:
            # Try loading the image
            pixmap = QPixmap(image_path)

            if pixmap.isNull():
                # If load fails, show an error message
                self.empty_state.setText("Failed to load image\n(Unsupported format or corrupted file)")
                self.stacked_widget.setCurrentIndex(0)
                return

            # Set image info
            filename = image_path.split("/")[-1]
            self.image_name.setText(f"File: {filename[:25]}{'...' if len(filename) > 25 else ''}")
            self.image_size.setText(f"Dimensions: {pixmap.width()}×{pixmap.height()}")

            # Scale to fit the preview area
            max_width = self.width() - 40
            max_height = 300

            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(
                    max_width, max_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

            # Create a bordered pixmap
            bordered_pixmap = QPixmap(pixmap.width() + 4, pixmap.height() + 4)
            bordered_pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(bordered_pixmap)
            painter.setPen(Qt.GlobalColor.darkGray)
            painter.drawRect(0, 0, pixmap.width() + 3, pixmap.height() + 3)
            painter.drawPixmap(2, 2, pixmap)
            painter.end()

            self.image_label.setPixmap(bordered_pixmap)
            self.stacked_widget.setCurrentIndex(1)

        except Exception as e:
            print(f"Error loading image: {e}")
            self.empty_state.setText(f"Error loading image: {str(e)}")
            self.stacked_widget.setCurrentIndex(0)

    def contextMenuEvent(self, event):

        cmenu = QMenu(self)
        pre_wsi = cmenu.addAction("Previous")
        next_wsi = cmenu.addAction("Next")
        patch_view = cmenu.addAction("Patch Preview")
        action = cmenu.exec(self.mapToGlobal(event.pos()))

        if action == pre_wsi:
            """"""
            self.parent_main.pre_wsi_img()
            # QMessageBox.about(self, 'Info', 'This Is Previous Wsi')
        elif action == next_wsi:
            """"""
            self.parent_main.next_wsi_img()
            # QMessageBox.about(self, 'Info', 'This Is Next Wsi')
        elif action == patch_view:
            """"""
            self.chile_Win = PatchPreviewWindow()
            self.chile_Win.show()


class MedicalAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.patient_id = None
        # Keep the most recent output directories for report generation and pipeline resumption
        self.last_cross_validation_dir = None
        self.last_degs_dir = None
        self.last_fea_gsea_dir = None
        self.signals = ToolSignals()
        self.signals.tool_end.connect(self.append_output, Qt.ConnectionType.QueuedConnection)
        self.signals.error_occurred.connect(self.append_output, Qt.ConnectionType.QueuedConnection)
        self.callback_handler = ToolCallbackHandler(self.signals)
        self.agent = WSIAgent(gui=self)
        self.setWindowTitle("WSI Analysis System")
        self.setGeometry(100, 100, 1200, 800)

        # Main splitter (left/right layout)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: chat interface
        chat_widget = self.create_chat_interface()

        # Right: operation panel
        operation_widget = self.create_operation_panel()

        main_splitter.addWidget(chat_widget)
        main_splitter.addWidget(operation_widget)
        main_splitter.setSizes([700, 500])  # initial ratio 7:5

        self.setCentralWidget(main_splitter)

        # Connect signals/slots
        self.connect_signals()

        self.worker_thread = QThread()
        self.worker = None

    def create_chat_interface(self):
        """Create the left chat interface."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Chat area
        chat_frame = QFrame()
        chat_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
            }
        """)
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Medical Assistant Chat")
        title_font = QFont("Arial", 14, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")

        # Chat history area
        self.chat_history = QListWidget()
        self.chat_history.setStyleSheet("""
            QListWidget {
                background-color: #f5f9fc;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            QListWidget::item {
                border: none;
                padding: 5px;
            }
            QListWidget::item:selected {
                background: none;
            }
        """)
        self.chat_history.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)

        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.setStyleSheet("""
            height: 30px;
            border-radius: 15px;
            padding: 5px 15px;
            background-color: #ffffff;
            border: 1px solid #bdbdbd;
        """)

        send_button = QPushButton("Send")
        send_button.setFixedSize(80, 40)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
        """)
        self.send_button = send_button

        input_layout.addWidget(self.chat_input, 8)
        input_layout.addWidget(send_button, 2)

        # Add to chat layout
        chat_layout.addWidget(title)
        chat_layout.addWidget(self.chat_history, 7)
        chat_layout.addLayout(input_layout, 1)

        # Add to main layout
        layout.addWidget(chat_frame, 1)  # chat area fills the left side

        widget.setLayout(layout)
        return widget

    def create_operation_panel(self):
        """Create the right operation panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Title
        title = QLabel("WSI Analysis Tools")
        title_font = QFont("Arial", 14, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c3e50;")

        # Add WSI preview widget to the top of the right panel
        self.preview_widget = PreviewWidget(self)
        # Keep the preview area compact so the upload section has enough space
        layout.addWidget(self.preview_widget, 1)

        # File upload section
        upload_group = QFrame()
        # Set a minimum height to prevent the upload section from being squeezed
        upload_group.setMinimumHeight(260)
        upload_group.setStyleSheet("""
            QFrame {
                background-color: #e8f4f8;
                border-radius: 10px;
                padding: 5px;
                border: 1px solid #d0e0e8;
            }
        """)
        upload_layout = QVBoxLayout()

        # WSI path upload
        wsi_layout = QHBoxLayout()
        wsi_label = QLabel("WSI Path")
        wsi_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        wsi_label.setFixedWidth(120)
        self.wsi_path = QLineEdit()
        self.wsi_path.setPlaceholderText("Select WSI file...")
        wsi_button = QPushButton("Select")
        wsi_button.clicked.connect(self.select_fold)
        wsi_button.setFixedWidth(80)
        wsi_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        wsi_layout.addWidget(wsi_label, 1)
        wsi_layout.addWidget(self.wsi_path, 5)
        wsi_layout.addWidget(wsi_button, 1)

        # Bulk RNA upload
        mask_layout = QHBoxLayout()
        mask_label = QLabel("RNA Path")
        mask_label.setFixedWidth(120)
        mask_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("Select mask file...")
        mask_button = QPushButton("Upload")
        mask_button.clicked.connect(self.upload_rna)
        mask_button.setFixedWidth(80)
        mask_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        mask_layout.addWidget(mask_label, 1)
        mask_layout.addWidget(self.mask_path, 5)
        mask_layout.addWidget(mask_button, 1)

        # Segmentation results upload (folder selection)
        seg_res_layout = QHBoxLayout()
        seg_res_label = QLabel("Seg Results")
        seg_res_label.setFixedWidth(120)
        seg_res_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        self.seg_results_path = QLineEdit()
        self.seg_results_path.setPlaceholderText("Select segmentation results folder (optional)...")
        seg_res_button = QPushButton("Select")
        seg_res_button.clicked.connect(self.select_seg_results_folder)
        seg_res_button.setFixedWidth(80)
        seg_res_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        seg_res_layout.addWidget(seg_res_label, 1)
        seg_res_layout.addWidget(self.seg_results_path, 5)
        seg_res_layout.addWidget(seg_res_button, 1)

        # Label/survival file upload
        label_layout = QHBoxLayout()
        label_label = QLabel("Label Path")
        label_label.setFixedWidth(120)
        label_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        self.label_path = QLineEdit()
        self.label_path.setPlaceholderText("Select label file...")
        label_button = QPushButton("Upload")
        label_button.clicked.connect(self.upload_label)
        label_button.setFixedWidth(80)
        label_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        label_layout.addWidget(label_label, 1)
        label_layout.addWidget(self.label_path, 5)
        label_layout.addWidget(label_button, 1)

        # Feature matrix upload (optional: upload dataset_feature_matrix / M_matrix directly)
        feature_matrix_layout = QHBoxLayout()
        feature_matrix_label = QLabel("Feature Matrix")
        feature_matrix_label.setFixedWidth(120)
        feature_matrix_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        self.feature_matrix_path = QLineEdit()
        self.feature_matrix_path.setPlaceholderText("Select feature matrix csv (optional)...")
        feature_matrix_button = QPushButton("Upload")
        feature_matrix_button.clicked.connect(self.upload_feature_matrix)
        feature_matrix_button.setFixedWidth(80)
        feature_matrix_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        feature_matrix_layout.addWidget(feature_matrix_label, 1)
        feature_matrix_layout.addWidget(self.feature_matrix_path, 5)
        feature_matrix_layout.addWidget(feature_matrix_button, 1)

        # Clinical information upload
        clinical_layout = QHBoxLayout()
        clinical_label = QLabel("Clinical")
        clinical_label.setFixedWidth(120)
        clinical_label.setStyleSheet("""
            QLabel {
                padding: 0px 0px 0px 15px;
            }
        """)
        self.clinical_path = QLineEdit()
        self.clinical_path.setPlaceholderText("Select Clinical information csv...")
        clinical_button = QPushButton("Upload")
        clinical_button.clicked.connect(self.upload_clinical)
        clinical_button.setFixedWidth(80)
        clinical_button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                border-radius: 5px;
                padding:5px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)

        clinical_layout.addWidget(clinical_label, 1)
        clinical_layout.addWidget(self.clinical_path, 5)
        clinical_layout.addWidget(clinical_button, 1)


        upload_layout.addLayout(wsi_layout)
        upload_layout.addLayout(mask_layout)
        upload_layout.addLayout(seg_res_layout)
        upload_layout.addLayout(label_layout)
        upload_layout.addLayout(feature_matrix_layout)
        upload_layout.addLayout(clinical_layout)

        upload_group.setLayout(upload_layout)

        # Tool buttons section
        buttons_group = QFrame()
        buttons_group.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        buttons_layout = QVBoxLayout(buttons_group)
        buttons_layout.setSpacing(10)

        # Create tool buttons
        # self.roi_button = self.create_tool_button("ROI Segmentor", "🔍")
        # self.histo_button = self.create_tool_button("Histomorphometries Extractor", "📊")
        # self.model_button = self.create_tool_button("Model Constructor", "🧠")
        # self.path_button = self.create_tool_button("Path-Genomics Analyzer", "🧬")
        # self.whole_button = self.create_tool_button("Whole Process Analyzer", "🧬")

        self.roi_button = self.create_tool_button("ROI Segmentor")
        self.histo_button = self.create_tool_button("Histomorphometries Extractor")
        self.model_button = self.create_tool_button("Model Constructor")
        self.path_button = self.create_tool_button("Path-Genomics Analyzer")
        self.whole_button = self.create_tool_button("Whole Process Analyzer")
        # self.setting_button = self.create_tool_button("Settings")

        self.report_button = self.create_tool_button("Generate Reports")
        # Enabled after a report is generated; opens the PDF directly
        self.view_pdf_button = self.create_tool_button("View PDF")
        self.view_pdf_button.setEnabled(False)
        self.last_report_pdf_path = None

        layout1 = QHBoxLayout()
        layout1.addWidget(self.roi_button)
        layout1.addWidget(self.histo_button)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.model_button)
        layout2.addWidget(self.path_button)

        layout3 = QHBoxLayout()
        layout3.addWidget(self.whole_button)
        layout3.addWidget(self.report_button)
        layout3.addWidget(self.view_pdf_button)

        buttons_layout.addLayout(layout1)
        buttons_layout.addLayout(layout2)
        buttons_layout.addLayout(layout3)
        # buttons_layout.addWidget(self.roi_button)
        # buttons_layout.addWidget(self.histo_button)
        # buttons_layout.addWidget(self.model_button)
        # buttons_layout.addWidget(self.path_button)
        # buttons_layout.addWidget(self.whole_button)

        # Runtime status (even if buttons are disabled, user can see what's running)
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        status_layout.setSpacing(6)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #2c3e50; font-weight: bold;")

        self.busy_bar = QProgressBar()
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setRange(0, 0)  # indeterminate progress: keep spinning
        self.busy_bar.setFixedHeight(10)
        self.busy_bar.setVisible(False)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.busy_bar)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.setFixedHeight(40)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)

        self.setting_button = QPushButton("Setting Params")
        self.setting_button.setFixedHeight(40)
        self.setting_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #219653;
            }
        """)

        bottom_layout.addWidget(self.clear_button)
        bottom_layout.addWidget(self.setting_button)
        self.setting_views = ParamsSettingWindow()

        # Add to main layout
        layout.addWidget(title)
        layout.addWidget(upload_group, 1)
        layout.addWidget(buttons_group, 3)
        layout.addWidget(status_frame, 0)
        layout.addLayout(bottom_layout, 1)

        widget.setLayout(layout)
        return widget

    def create_tool_button(self, text, icon=None):
        """Create a tool button with unified styling."""
        button = QPushButton(text)
        button.setFixedHeight(40)

        if icon:
            button.setText(f"{icon} {text}")

        button.setStyleSheet("""
            QPushButton {
                background-color: #5dade2;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                font-size: 13px;
                text-align: left;
                padding-left: 20px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #2c81ba;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #f5f5f5;
            }
        """)
        return button

    def connect_signals(self):
        """Connect signals and slots."""
        self.send_button.clicked.connect(self.send_message)
        self.chat_input.returnPressed.connect(self.send_message)  # press Enter to send
        self.clear_button.clicked.connect(self.clear_chat)
        self.report_button.clicked.connect(self.send_default_msg)
        self.view_pdf_button.clicked.connect(self.open_last_report_pdf)
        self.roi_button.clicked.connect(self.send_default_msg)
        self.histo_button.clicked.connect(self.send_default_msg)
        self.model_button.clicked.connect(self.send_default_msg)
        self.path_button.clicked.connect(self.send_default_msg)
        self.whole_button.clicked.connect(self.send_default_msg)

        self.setting_button.clicked.connect(self.show_setting)

    def show_setting(self):
        self.setting_views.show()

    def _build_agent_context(self, user_text: str, extra_overrides: Optional[dict] = None) -> dict:
        """
        Build a machine-readable context payload from current UI fields + Setting Params,
        so the agent can call tools without asking for missing arguments.
        """
        sender_text = ""
        try:
            sender = self.sender()
            sender_text = sender.text() if sender else ""
        except Exception:
            sender_text = ""

        try:
            setting_params = self.setting_views.get_val()
        except Exception:
            setting_params = {}

        # Current UI paths (may be empty strings)
        ui_paths = {
            "wsi_folder": getattr(self, "wsi_path", None).text().strip() if getattr(self, "wsi_path", None) else "",
            "seg_results_folder": getattr(self, "seg_results_path", None).text().strip() if getattr(self, "seg_results_path", None) else "",
            "label_path": getattr(self, "label_path", None).text().strip() if getattr(self, "label_path", None) else "",
            "feature_matrix_csv": getattr(self, "feature_matrix_path", None).text().strip() if getattr(self, "feature_matrix_path", None) else "",
            "bulk_rna_csv": getattr(self, "mask_path", None).text().strip() if getattr(self, "mask_path", None) else "",
            "clinical_info_csv": getattr(self, "clinical_path", None).text().strip() if getattr(self, "clinical_path", None) else "",
        }

        # Optional: infer intent-based overrides from the user's text
        overrides = {}
        lt = (user_text or "").lower()
        if "texture" in lt:
            overrides.setdefault("extract_features", {})
            overrides["extract_features"]["feature_type"] = "texture"
        if "end-to-end" in lt or "end to end" in lt or "pipeline" in lt:
            # Prefer running the full pipeline end-to-end
            overrides["preferred_flow"] = "whole_pipeline"

        # If the prompt is about prognosis/modeling only (no transcriptomics/pathway keywords),
        # limit the pipeline to model construction + cross-validation.
        transcriptomics_markers = (
            "transcript", "transcriptomic", "expression profile", "expression",
            "rna", "bulk", "deg", "gsea", "enrichment", "pathway", "molecular driver",
        )
        has_transcriptomics_intent = any(k in lt for k in transcriptomics_markers)
        if (("end-to-end" in lt or "end to end" in lt or "pipeline" in lt) and (not has_transcriptomics_intent)):
            overrides["limit_to_modeling"] = True
        if extra_overrides:
            try:
                overrides.update(extra_overrides)
            except Exception:
                pass

        return {
            "sender_button": sender_text,
            "patient_id": getattr(self, "patient_id", None),
            "ui_paths": ui_paths,
            "setting_params": setting_params,
            "overrides": overrides,
            "outputs": {
                "last_cross_validation_dir": getattr(self, "last_cross_validation_dir", None) or "",
                "last_degs_dir": getattr(self, "last_degs_dir", None),
                "last_fea_gsea_dir": getattr(self, "last_fea_gsea_dir", None),
                "last_report_pdf_path": getattr(self, "last_report_pdf_path", None),
            },
        }

    def _augment_user_message(self, user_text: str, extra_overrides: Optional[dict] = None) -> str:
        """
        Append CONTEXT_JSON to the user's message so the agent can reliably fill tool arguments.
        """
        ctx = self._build_agent_context(user_text, extra_overrides=extra_overrides)
        # Keep the original user text intact; append context in a consistent marker block.
        return f"{user_text}\n\nCONTEXT_JSON:\n{json.dumps(ctx, indent=2)}\n"

    def _needs_segmentation_prompt(self, user_text: str) -> bool:
        lt = (user_text or "").lower()
        return any(k in lt for k in ("segment", "segmentation", "end-to-end", "end to end", "pipeline"))

    def _needs_extraction_prompt(self, user_text: str) -> bool:
        lt = (user_text or "").lower()
        return any(k in lt for k in ("extract", "feature extraction", "end-to-end", "end to end", "pipeline", "model", "cross-validation", "cross validation"))

    def _prompt_existing_seg_results(self) -> Optional[str]:
        """Return one of: rerun, skip, preview, or None if dialog closed."""
        seg_folder = getattr(self, "seg_results_path", None).text().strip() if getattr(self, "seg_results_path", None) else ""
        if not seg_folder:
            return None

        box = QMessageBox(self)
        box.setWindowTitle("Segmentation task")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText('Detected that you have already selected a \"Segmentation Results\" folder. What would you like to do?')
        box.setInformativeText("You can rerun segmentation, skip segmentation and proceed to the next step, or preview the results first.")
        btn_reseg = box.addButton("Rerun segmentation", QMessageBox.ButtonRole.AcceptRole)
        btn_skip = box.addButton("Skip segmentation (next step)", QMessageBox.ButtonRole.DestructiveRole)
        btn_preview = box.addButton("Preview results", QMessageBox.ButtonRole.ActionRole)
        bbox = box.findChild(QDialogButtonBox)
        if bbox:
            bbox.setCenterButtons(True)
        box.exec()

        clicked = box.clickedButton()
        if clicked is None:
            return None
        if clicked == btn_preview:
            return "preview"
        if clicked == btn_skip:
            return "skip"
        return "rerun"

    def _prompt_existing_feature_matrix(self) -> Optional[str]:
        """Return one of: rerun, skip, or None if dialog closed."""
        feat_csv = getattr(self, "feature_matrix_path", None).text().strip() if getattr(self, "feature_matrix_path", None) else ""
        if not feat_csv:
            return None

        box = QMessageBox(self)
        box.setWindowTitle("Feature extraction task")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText('Detected that you have already provided a \"Feature Matrix\". What would you like to do?')
        box.setInformativeText("If you already have a feature matrix, you can skip feature extraction and proceed to model construction.")
        btn_reextract = box.addButton("Rerun feature extraction", QMessageBox.ButtonRole.AcceptRole)
        btn_skip = box.addButton("Skip feature extraction (next step)", QMessageBox.ButtonRole.DestructiveRole)
        bbox = box.findChild(QDialogButtonBox)
        if bbox:
            bbox.setCenterButtons(True)
        box.exec()

        clicked = box.clickedButton()
        if clicked is None:
            return None
        if clicked == btn_skip:
            return "skip"
        if clicked == btn_reextract:
            return "rerun"
        return None

    def _prompt_continue_after_preview(self) -> bool:
        """
        After opening the browser preview, ask whether to continue running the pipeline.
        Returns True if user wants to continue; False if user wants to stop.
        """
        box = QMessageBox(self)
        box.setWindowTitle("Continue after preview?")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText("Preview is open in your browser. Do you want to continue running the pipeline?")
        btn_continue = box.addButton("Continue pipeline", QMessageBox.ButtonRole.AcceptRole)
        btn_stop = box.addButton("Stop", QMessageBox.ButtonRole.DestructiveRole)
        bbox = box.findChild(QDialogButtonBox)
        if bbox:
            bbox.setCenterButtons(True)
        box.exec()

        clicked = box.clickedButton()
        if clicked == btn_continue:
            return True
        if clicked == btn_stop:
            return False
        return False

    def _status_for_chat_prompt(self, user_text: str) -> str:
        """
        Infer a more informative status label for chat-triggered runs.
        This only affects the status bar text; it does not change tool-routing logic.
        """
        lt = (user_text or "").lower()

        # Rough intent detection
        wants_report = "report" in lt or "pdf" in lt
        wants_genomics = any(k in lt for k in ("transcript", "transcriptomic", "rna", "deg", "gsea", "enrichment", "pathway"))
        wants_model = any(k in lt for k in ("model", "predict", "prognos", "cross-validation", "cross validation", "feature selection"))
        wants_extract = any(k in lt for k in ("extract", "feature", "texture", "morph", "radiomic"))
        wants_segment = any(k in lt for k in ("segment", "segmentation", "tumor", "cell"))
        wants_pipeline = any(k in lt for k in ("end-to-end", "end to end", "pipeline", "end-to-end pipeline"))

        if wants_pipeline:
            return "Status: Running (Whole pipeline, chat)..."
        if wants_report:
            return "Status: Running (Generate Reports, chat)..."
        if wants_genomics:
            return "Status: Running (Path-Genomics Analyzer, chat)..."
        if wants_model:
            return "Status: Running (Model Constructor, chat)..."
        if wants_extract and wants_segment:
            return "Status: Running (Segmentation + Feature Extraction, chat)..."
        if wants_extract:
            return "Status: Running (Histomorphometries Extractor, chat)..."
        if wants_segment:
            return "Status: Running (ROI Segmentor, chat)..."
        return "Status: Processing (chat)..."

    def send_message(self):
        """Send a chat message."""
        message = self.chat_input.text().strip()
        if message:
            # Add user message (left)
            self.add_message(message, is_ai=False)
            self.chat_input.clear()

            # Show a more informative status for chat-triggered runs
            self.off_use(self._status_for_chat_prompt(message))
            QApplication.processEvents()  # update UI immediately

            # Trigger agent response (simulated processing)
            extra_overrides: dict = {}

            # Keep existing GUI UX: if user already selected outputs, prompt before rerunning steps.
            if getattr(self, "seg_results_path", None) and self.seg_results_path.text().strip() and self._needs_segmentation_prompt(message):
                decision = self._prompt_existing_seg_results()
                if decision is None:
                    # Dialog closed -> do nothing
                    self.on_agent_finished()
                    return
                if decision == "preview":
                    QDesktopServices.openUrl(QUrl("http://127.0.0.1:8366/"))
                    self.add_message("Opened the segmentation results preview in your browser.", is_ai=True)
                    if not self._prompt_continue_after_preview():
                        self.on_agent_finished()
                        return
                    # Continue: treat segmentation as already done (use existing results)
                    extra_overrides["skip_segmentation"] = True
                if decision == "skip":
                    extra_overrides["skip_segmentation"] = True

            if getattr(self, "feature_matrix_path", None) and self.feature_matrix_path.text().strip() and self._needs_extraction_prompt(message):
                decision = self._prompt_existing_feature_matrix()
                if decision is None:
                    self.on_agent_finished()
                    return
                if decision == "skip":
                    extra_overrides["skip_feature_extraction"] = True

            # Set LLM configuration before running the agent
            try:
                from wsi_tools import set_llm_config
                llm_config = self.setting_views.get_val().get("llm_config", {})
                set_llm_config(
                    api_key=llm_config.get("api_key"),
                    base_url=llm_config.get("base_url"),
                    report_model=llm_config.get("report_model")
                )
            except Exception as e:
                print(f"Warning: Failed to set LLM config: {e}")

            self.simulate_ai_response(self._augment_user_message(message, extra_overrides=extra_overrides))

    def add_message(self, message, is_ai=False):
        """Add a message to chat history."""
        # Create bubble widget
        bubble = ChatBubble(message, is_ai)
        # Create list item
        item = QListWidgetItem()
        item.setSizeHint(bubble.getSizeHint())
        # Add to list
        self.chat_history.addItem(item)
        self.chat_history.setItemWidget(item, bubble)
        # Scroll to bottom
        self.chat_history.scrollToBottom()

    def add_ai_message(self, message):
        """Add an AI message to chat history."""
        self.add_message(message, is_ai=True)

    def append_output(self, text):
        # Ensure UI updates happen on the UI thread
        # Extract the PDF path from output for the "View PDF" button
        try:
            self._maybe_capture_report_pdf_path(text)
            self._maybe_capture_analysis_dirs(text)
        except Exception:
            # Convenience only; do not affect the main flow
            pass

        # Update status with the current sub-task based on runtime logs
        try:
            if hasattr(self, "status_label") and hasattr(self, "busy_bar") and text:
                t = str(text)
                t_low = t.lower()

                # Model construction / cross-validation
                if any(k in t for k in ("Building model", "Remove correlated features", "Correlation threshold", "k_fold", "repeats_num")):
                    self.status_label.setText("Status: Running (Model construction + cross-validation)...")
                    self.busy_bar.setVisible(True)
                    QApplication.processEvents()
                # Segmentation
                elif "Running segmentation" in t or "segment_image:" in t:
                    self.status_label.setText("Status: Running (Segmentation)...")
                    self.busy_bar.setVisible(True)
                    QApplication.processEvents()
                # Feature extraction / aggregation
                elif "Extracting features" in t or "Feature extraction and aggregation completed" in t:
                    self.status_label.setText("Status: Running (Feature extraction + aggregation)...")
                    self.busy_bar.setVisible(True)
                    QApplication.processEvents()
                # Transcriptomics association (DEG/GSEA)
                elif any(k in t_low for k in ("extracting gene data", "performing de analysis", "enrichment_analysis", "gsea", "deg results", "enrichment results")):
                    self.status_label.setText("Status: Running (Path-Genomics Analyzer: DEG + enrichment/GSEA)...")
                    self.busy_bar.setVisible(True)
                    QApplication.processEvents()
                # Report generation
                elif "Generating report" in t or "Report generation failed" in t or "PDF report saved to:" in t:
                    self.status_label.setText("Status: Running (Generate Reports)...")
                    self.busy_bar.setVisible(True)
                    QApplication.processEvents()
        except Exception:
            pass

        # Make status updates more responsive: if we detect "done/saved path" outputs,
        # update status early (instead of waiting for the worker thread to exit).
        try:
            if hasattr(self, "status_label") and hasattr(self, "busy_bar") and text:
                done_markers = (
                    # English outputs (wsi_tools.py)
                    "Model outputs saved to:",
                    "DEG results:",
                    "Enrichment results:",
                    "PDF report saved to:",
                    "finished. the results have been saved",
                )
                if any(m in text for m in done_markers):
                    self.status_label.setText("Status: Completed (updating UI...)")
                    # Hide the progress bar to make completion obvious
                    self.busy_bar.setVisible(False)
                    QApplication.processEvents()
        except Exception:
            pass

        event = AppendOutputEvent(text)
        QApplication.postEvent(self, event)

    def event(self, event):
        if isinstance(event, AppendOutputEvent):
            self.add_message(event.text, is_ai=True)
            return True
        return super().event(event)

    def run_agent(self, user_message):
        """"""
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker = AgentWorker(
            self.agent,
            user_message,
            self.signals
        )

        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.on_agent_finished)
        self.worker_thread.finished.connect(self.worker.deleteLater)

        self.worker_thread.start()

    def simulate_ai_response(self, user_message):
        """Run agent in worker thread."""
        self.run_agent(user_message)


    def on_agent_finished(self):
        self.chat_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.roi_button.setEnabled(True)
        self.histo_button.setEnabled(True)
        self.model_button.setEnabled(True)
        self.path_button.setEnabled(True)
        self.whole_button.setEnabled(True)
        self.report_button.setEnabled(True)
        # Enable "View PDF" if a PDF path is available
        self.view_pdf_button.setEnabled(bool(self.last_report_pdf_path))
        self.clear_button.setEnabled(True)
        if hasattr(self, "busy_bar"):
            self.busy_bar.setVisible(False)
        if hasattr(self, "status_label"):
            self.status_label.setText("Status: Ready")
        # Force one UI refresh to reduce perceived lag after completion
        try:
            QApplication.processEvents()
        except Exception:
            pass
        self.chat_input.setFocus()

    def off_use(self, status_text: str = "Status: Running..."):
        """"""
        self.chat_input.setEnabled(False)
        self.send_button.setEnabled(False)
        self.roi_button.setEnabled(False)
        self.histo_button.setEnabled(False)
        self.model_button.setEnabled(False)
        self.path_button.setEnabled(False)
        self.whole_button.setEnabled(False)
        self.report_button.setEnabled(False)
        self.view_pdf_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        if hasattr(self, "status_label") and status_text:
            self.status_label.setText(status_text)
        if hasattr(self, "busy_bar"):
            self.busy_bar.setVisible(True)

    def _maybe_capture_report_pdf_path(self, text: str) -> None:
        """
        Capture the PDF path from agent output.

        wsi_tools.generate_report returns a line like:
        "PDF report saved to: {pdf_path}"
        """
        if not text:
            return
        m = re.search(r"PDF report saved to:\s*([^\r\n`\"']+?\.pdf)", text)
        if not m:
            return
        pdf_path = m.group(1).strip().strip("`").strip().strip('"').strip("'")
        if not pdf_path:
            return
        self.last_report_pdf_path = pdf_path
        self.view_pdf_button.setEnabled(True)

    def _maybe_capture_analysis_dirs(self, text: str) -> None:
        """
        Capture DEG / enrichment output directories from analyze_results output,
        and cross_validation output directory from build_model output,
        so "Generate Reports" can pass explicit parameters.

        Expected outputs:
        - "Model outputs saved to: xxx/cross_validation"
        - "DEG results: xxx; Enrichment results: yyy"
        """
        if not text:
            return
        # Capture cross_validation directory from build_model
        m0 = re.search(r"Model outputs saved to:\s*([^\r\n]+)", text)
        if m0:
            cv_path = m0.group(1).strip().strip("`").strip('"').strip("'")
            if cv_path and "cross_validation" in cv_path:
                self.last_cross_validation_dir = cv_path
        # Capture DEG / enrichment directories from analyze_results
        m1 = re.search(r"DEG results:\s*([^\r\n;]+)", text)
        m2 = re.search(r"Enrichment results:\s*([^\r\n;]+)", text)
        if m1:
            self.last_degs_dir = m1.group(1).strip().strip("`")
        if m2:
            self.last_fea_gsea_dir = m2.group(1).strip().strip("`")

    def open_last_report_pdf(self):
        """Open the most recently generated PDF report with the default system app."""
        pdf_path = self.last_report_pdf_path
        if not pdf_path:
            QMessageBox.information(self, "Info", "No PDF report path detected yet. Please generate a report first.")
            return
        p = Path(pdf_path)
        if not p.exists():
            QMessageBox.warning(self, "Warning", f"PDF file does not exist:\n{p}\n\nPlease confirm the report was generated successfully, or try again later.")
            return
        url = QUrl.fromLocalFile(str(p.resolve()))
        ok = QDesktopServices.openUrl(url)
        if not ok and os.name == "nt":
            try:
                os.startfile(str(p.resolve()))  # noqa: S606
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to open PDF: {e}")

    def clear_chat(self):
        """Clear chat history."""
        self.chat_history.clear()
        self.agent.delete_memory()
        # Add welcome message
        self.add_message("Welcome to the WSI Analysis System! How can I assist you today?", is_ai=True)

    def generate_report(self):
        """Generate a report."""
        # Add a system message
        self.add_message("Generate an analysis report for the latest dataset.", is_ai=True)

    # Legacy send_default_msg implementation was removed to keep the file clean and English-only.

    def send_default_msg(self):
        """"""
        setting_params = self.setting_views.get_val()
        sender = self.sender()
        is_ai = True
        sender_text = sender.text() if sender else ""

        # UX enhancement: if the user already provided segmentation results / feature matrix,
        # show options before running the step again.
        # 1) Segmentation: if a segmentation results folder is selected, offer (rerun / skip / preview)
        if "Segmentor" in sender_text:
            seg_folder = ""
            try:
                seg_folder = getattr(self, "seg_results_path", None).text().strip()
            except Exception:
                seg_folder = ""
            if seg_folder:
                box = QMessageBox(self)
                box.setWindowTitle("Segmentation task")
                box.setIcon(QMessageBox.Icon.Question)
                box.setText('Detected that you have already selected a "Segmentation Results" folder. What would you like to do?')
                box.setInformativeText(
                    "You can rerun segmentation, skip segmentation and proceed to the next step, or preview the results first."
                )
                btn_reseg = box.addButton("Rerun segmentation", QMessageBox.ButtonRole.AcceptRole)
                btn_skip = box.addButton("Skip segmentation (next step)", QMessageBox.ButtonRole.DestructiveRole)
                btn_preview = box.addButton("Preview results", QMessageBox.ButtonRole.ActionRole)
                bbox = box.findChild(QDialogButtonBox)
                if bbox:
                    bbox.setCenterButtons(True)
                box.exec()

                clicked = box.clickedButton()
                if clicked == btn_preview:
                    # Browser preview (local service)
                    QDesktopServices.openUrl(QUrl("http://127.0.0.1:8366/"))
                    self.add_message("Opened the segmentation results preview in your browser.", is_ai=True)
                    return
                if clicked == btn_skip:
                    self.add_message(
                        'Segmentation skipped. You can now click "Histomorphometries Extractor" to extract features.',
                        is_ai=True,
                    )
                    return
                # btn_reseg: continue with the normal flow (run segmentation)

        # 2) Feature extraction: if a feature matrix is uploaded, offer (rerun / skip to next step)
        if "Extractor" in sender_text:
            feat_csv = ""
            try:
                feat_csv = getattr(self, "feature_matrix_path", None).text().strip()
            except Exception:
                feat_csv = ""
            if feat_csv:
                box = QMessageBox(self)
                box.setWindowTitle("Feature extraction task")
                box.setIcon(QMessageBox.Icon.Question)
                box.setText('Detected that you have already provided a "Feature Matrix". What would you like to do?')
                box.setInformativeText(
                    "If you already have a feature matrix, you can skip feature extraction and proceed to model construction."
                )
                btn_reextract = box.addButton("Rerun feature extraction", QMessageBox.ButtonRole.AcceptRole)
                btn_skip = box.addButton("Skip feature extraction (next step)", QMessageBox.ButtonRole.DestructiveRole)
                bbox = box.findChild(QDialogButtonBox)
                if bbox:
                    bbox.setCenterButtons(True)
                box.exec()

                clicked = box.clickedButton()
                if clicked == btn_skip:
                    self.add_message(
                        'Feature extraction skipped. You can now click "Model Constructor" (the uploaded Feature Matrix will be used).',
                        is_ai=True,
                    )
                    return
                if clicked != btn_reextract:
                    return
                # btn_reextract: continue with the normal flow (run feature extraction)
        if "Segmentor" in sender.text():
            if self.wsi_path.text():
                self.patient_id = f"{int(time.time() * 1000)}"
                seg_params = setting_params['segment_image']
                seg_params['patient_id'] = "patient_" + self.patient_id
                message = (
                    f"Run segmentation on the WSI folder:\n{self.wsi_path.text()}\n\n"
                    f"Default parameters:\n{json.dumps(seg_params)}"
                )
            else:
                message = "Please select a WSI folder to segment."
                is_ai = False
        elif "Extractor" in sender.text():
            if self.label_path.text():
                message = (
                    f"Extract features based on segmentation results.\n"
                    f"Survival/label file: {self.label_path.text()}\n\n"
                    f"Default parameters:\n{json.dumps(setting_params['extract_features'])}"
                )
            else:
                message = "Please select a survival/label file (CSV)."
                is_ai = False
        elif "Model" in sender.text():
            if not self.label_path.text():
                message = "Please select a survival/label file (csv/xlsx/xls)."
                is_ai = False
            else:
                # If the user didn't run Segmentor/Whole first, auto-generate patient_id
                # to avoid required tool parameters blocking execution.
                if not getattr(self, "patient_id", None):
                    self.patient_id = f"{int(time.time() * 1000)}"

                model_params = dict(setting_params["build_model"])
                model_params["patient_id"] = "patient_" + self.patient_id
                model_params["label_path"] = self.label_path.text()

                # extract_path: prefer user-uploaded feature matrix CSV
                if getattr(self, "feature_matrix_path", None) and self.feature_matrix_path.text().strip():
                    model_params["extract_path"] = self.feature_matrix_path.text().strip()
                else:
                    # Otherwise use the default convention: example_folder/patient_xxx/aggregation/
                    agg_dir = (_REPO_ROOT / "example_folder" / ("patient_" + self.patient_id) / "aggregation").resolve()
                    model_params["extract_path"] = str(agg_dir)

                message = (
                    "Model construction and cross-validation using the feature matrix and survival and label information file (will call build_model).\n"
                    f"Parameters:\n{json.dumps(model_params)}"
                )
        elif "Genomics" in sender.text():
            if self.mask_path.text() and self.clinical_path.text():
                if setting_params['analyze_results'].get("clinical_col"):
                    # Auto-fill required analyze_results parameters
                    if not getattr(self, "patient_id", None):
                        self.patient_id = f"{int(time.time() * 1000)}"
                    pid = "patient_" + self.patient_id

                    gene_params = dict(setting_params["analyze_results"])
                    gene_params["patient_id"] = pid
                    gene_params["bulk_rna_path"] = self.mask_path.text()
                    gene_params["clinical_info_path"] = self.clinical_path.text()

                    # cross_validation_path: default to the current patient's output directory
                    cross_dir = (_REPO_ROOT / "example_folder" / pid / "cross_validation").resolve()
                    gene_params["cross_validation_path"] = str(cross_dir)

                    # extract_path: prefer user-uploaded feature matrix CSV; otherwise use aggregation dir
                    if getattr(self, "feature_matrix_path", None) and self.feature_matrix_path.text().strip():
                        gene_params["extract_path"] = self.feature_matrix_path.text().strip()
                    else:
                        agg_dir = (_REPO_ROOT / "example_folder" / pid / "aggregation").resolve()
                        gene_params["extract_path"] = str(agg_dir)

                    message = (
                        "Run transcriptomics association analysis based on model results (DEG + enrichment/GSEA).\n"
                        f"Parameters:\n{json.dumps(gene_params)}"
                    )
                else:
                    is_ai = False
                    message = "Please configure clinical_col in Setting Params."
            else:
                message = "Please select a bulk RNA CSV file and a clinical_information CSV file."
                is_ai = False
        elif "Report" in sender.text():
            # Prefer structured parameters to avoid missing tool arguments
            if self.last_fea_gsea_dir:
                payload = {
                    "gene_results_fold_path": self.last_fea_gsea_dir,
                    "differential_results_fold_path": self.last_degs_dir or "",
                }
                message = f"Generate the pathology report (will call generate_report).\nParameters:\n{json.dumps(payload)}"
            else:
                message = "Generate the pathology report based on the previous step (run Path-Genomics Analyzer first to produce FEA_GSEA outputs)."
        elif "Whole" in sender.text():
            self.patient_id = f"{int(time.time() * 1000)}"
            seg_params = setting_params['segment_image']
            seg_params['patient_id'] = "patient_" + self.patient_id
            if self.wsi_path.text() and self.label_path.text() and self.mask_path.text() and self.clinical_path.text():
                if setting_params['analyze_results'].get("clinical_col"):
                    message = (
                        "Run the full pipeline with the provided inputs.\n\n"
                        f"WSI folder: {self.wsi_path.text()}\n"
                        f"Survival/label file: {self.label_path.text()}\n"
                        f"Bulk RNA file: {self.mask_path.text()}\n"
                        f"Clinical information CSV: {self.clinical_path.text()}\n\n"
                        "Pipeline order: 1) Segmentation -> 2) Feature extraction -> 3) Model construction -> "
                        "4) Result analysis -> 5) Report generation\n\n"
                        f"Segmentation params: {json.dumps(seg_params)}\n"
                        f"Feature extraction params: {json.dumps(setting_params['extract_features'])}\n"
                        f"Model construction params: {json.dumps(setting_params['build_model'])}\n"
                        f"Result analysis params: {json.dumps(setting_params['analyze_results'])}"
                    )
                else:
                    is_ai = False
                    message = "Please configure clinical_col in Setting Params."
            else:
                message = "Please select a bulk RNA CSV file and a clinical_information CSV file."
                is_ai = False
        else:
            message = "Please select an action."
            is_ai = False
        if message:
            self.add_message(message)
            QApplication.processEvents()
            if is_ai:
                # Make it explicit which step is currently running
                if "Segmentor" in sender_text:
                    status = "Status: Running (ROI Segmentor)..."
                elif "Extractor" in sender_text:
                    status = "Status: Running (Histomorphometries Extractor)..."
                elif "Model" in sender_text:
                    status = "Status: Running (Model Constructor)..."
                elif "Genomics" in sender_text:
                    status = "Status: Running (Path-Genomics Analyzer)..."
                elif "Whole" in sender_text:
                    status = "Status: Running (Whole Process Analyzer)..."
                elif "Report" in sender_text or "Generate Reports" in sender_text:
                    status = "Status: Running (Generate Reports)..."
                else:
                    status = "Status: Running..."
                self.off_use(status)
                self.simulate_ai_response(message)

    def open_file(self):
        return QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*)"
        )

    def upload_rna(self):
        """"""
        file_path, _ = self.open_file()
        self.mask_path.setText(file_path)
        self.add_message(f"BulkRNA file selected: {file_path}", is_ai=True)

    def upload_label(self):
        """"""
        file_path, _ = self.open_file()
        self.label_path.setText(file_path)
        self.add_message(f"Label file selected: {file_path}", is_ai=True)

    def upload_feature_matrix(self):
        """"""
        file_path, _ = self.open_file()
        self.feature_matrix_path.setText(file_path)
        self.add_message(f"Feature matrix file selected: {file_path}", is_ai=True)

    def select_seg_results_folder(self):
        """
        Select a segmentation results folder (same UX as selecting a WSI folder).
        After selection, list all immediate subfolder names in the chat area.
        """
        start_dir = self.seg_results_path.text().strip() or str((_REPO_ROOT / "example_folder").resolve())
        folder = QFileDialog.getExistingDirectory(self, "Select Segmentation Results Folder", start_dir)
        if not folder:
            return
        self.seg_results_path.setText(folder)

        # List immediate subfolder names (no extra preview dialog)
        subdirs = []
        try:
            p = Path(folder).resolve()
            if p.exists() and p.is_dir():
                subdirs = sorted([d.name for d in p.iterdir() if d.is_dir()])
        except Exception:
            subdirs = []

        if subdirs:
            preview = "\n".join([f"- {name}" for name in subdirs])
            self.add_message(
                f"Segmentation results folder selected: {folder}\nSubfolders:\n{preview}",
                is_ai=True,
            )
        else:
            self.add_message(f"Segmentation results folder selected: {folder}\nSubfolders: (none)", is_ai=True)

    def upload_clinical(self):
        file_path, _ = self.open_file()
        self.clinical_path.setText(file_path)
        self.add_message(f"Clinical file selected: {file_path}", is_ai=True)

        # Auto-read clinical CSV header and populate Setting Params -> clinical_col dropdown
        cols = []
        try:
            if file_path and file_path.lower().endswith(".csv"):
                with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
                    sample = f.read(4096)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
                    except Exception:
                        dialect = csv.excel
                    reader = csv.reader(f, dialect)
                    header = next(reader, [])
                cols = [c.strip() for c in header if isinstance(c, str) and c.strip()]
        except Exception:
            cols = []

        try:
            self.setting_views.update_clinical_columns(cols)
        except Exception:
            # Do not affect the main flow
            pass

    def select_fold(self):
        """Handle selecting a WSI folder."""
        fold_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            ""
        )
        # self.fold_path = fold_path
        self.wsi_index = 0
        self.wsi_path.setText(fold_path)
        self.add_message(f"WSI file selected: {fold_path}", is_ai=True)
        swi_list = self.get_swi_list()
        if len(swi_list) > 0:
            file_path = get_swi_thumbnail(swi_list[self.wsi_index])
            self.preview_widget.set_preview_image(file_path)

    def get_swi_list(self):
        res = []
        if self.wsi_path.text():
            for suffix in ['*.svs', '*.tif', '*.tiff', '*.ndpi']:
                res += glob.glob(f'{self.wsi_path.text()}/{suffix}')
        return res

    def pre_wsi_img(self):
        """"""
        swi_list = self.get_swi_list()
        if len(swi_list) > 0:
            if self.wsi_index == 0:
                self.wsi_index = len(swi_list) - 1
            else:
                self.wsi_index -= 1
            file_path = get_swi_thumbnail(swi_list[self.wsi_index])
            self.preview_widget.set_preview_image(file_path)

    def next_wsi_img(self):
        """"""
        swi_list = self.get_swi_list()
        if len(swi_list) > 0:
            if self.wsi_index == len(swi_list) - 1:
                self.wsi_index = 0
            else:
                self.wsi_index += 1
            file_path = get_swi_thumbnail(swi_list[self.wsi_index])
            self.preview_widget.set_preview_image(file_path)



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Application stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #333333;
        }
        QLabel {
            font-size: 13px;
        }
        QLineEdit {
            font-size: 13px;
            padding: 5px 10px;
            border: 1px solid #cccccc;
            border-radius: 4px;
            background-color: white;
        }
        QListWidget {
            border: 1px solid #cccccc;
            background-color: white;
        }
        QFrame {
            border-radius: 10px;
        }
    """)

    window = MedicalAnalysisApp()

    # Add welcome message
    window.add_message("Welcome to the WSI Analysis System! I'm your AI assistant. How can I help you today?",
                       is_ai=True)

    window.show()
    sys.exit(app.exec())