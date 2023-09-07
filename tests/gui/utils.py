from pathlib import Path

from pyctbgui.utils.defines import Defines


class defaultParams:
    image = True
    chip = "Matterhorn"
    numpy = True
    mode = "Transceiver"
    outputDir = "/tmp"
    filename = "run"
    nFrames = 1
    enabled = []
    plotted = []
    pedestalRecord = True


def setup_gui(qtbot, widget, params=defaultParams(), tmp_path=Path('/tmp')):
    if params.image:
        widget.plotTab.view.radioButtonImage.setChecked(True)
        widget.plotTab.view.radioButtonWaveform.setChecked(False)
    else:
        widget.plotTab.view.radioButtonWaveform.setChecked(True)
        widget.plotTab.view.radioButtonImage.setChecked(False)
        if params.mode == "Transceiver":
            widget.tabWidget.setCurrentIndex(Defines.transceiver.tabIndex)

            for i in range(Defines.transceiver.count):
                # check or uncheck enable/plot checkboxes
                enable = getattr(widget.transceiverTab.view, f"checkBoxTransceiver{i}")
                enable.setChecked(i in params.enabled)
                enable = getattr(widget.transceiverTab.view, f"checkBoxTransceiver{i}Plot")
                enable.setChecked(i in params.plotted)

    qtbot.keyClicks(widget.plotTab.view.comboBoxPlot, params.chip)
    widget.acquisitionTab.view.checkBoxFileWriteNumpy.setChecked(params.numpy)
    qtbot.keyClicks(widget.acquisitionTab.view.comboBoxROMode, params.mode)

    widget.acquisitionTab.view.lineEditFilePath.setText(str(tmp_path))
    widget.acquisitionTab.setFilePath()

    widget.acquisitionTab.view.lineEditFileName.setText(params.filename)
    widget.acquisitionTab.setFileName()

    widget.acquisitionTab.view.spinBoxFrames.setValue(params.nFrames)
    widget.acquisitionTab.setFrames()

    widget.plotTab.view.radioButtonPedestalRecord.setChecked(params.pedestalRecord)
    widget.plotTab.view.radioButtonPedestalApply.setChecked(not params.pedestalRecord)

    qtbot.wait_until(lambda: widget.acquisitionTab.view.spinBoxFrames.value() == params.nFrames)

    assert widget.acquisitionTab.view.comboBoxROMode.currentText() == params.mode
    assert widget.plotTab.view.comboBoxPlot.currentText() == params.chip
    assert widget.acquisitionTab.writeNumpy == params.numpy
    assert widget.acquisitionTab.view.spinBoxFrames.value() == params.nFrames
    assert widget.acquisitionTab.outputDir == Path(str(tmp_path))
