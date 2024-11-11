import sys
import json
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from collections import deque
import threading
import time
from enum import Enum


class Quantization(Enum):
    WHOLE = 2.0  # Whole note duration in seconds
    HALF = 1.0  # Half note duration in seconds
    QUARTER = 0.5  # Quarter note duration in seconds


class FourierSynthesizer(QtWidgets.QWidget):
    """
    A comprehensive Fourier Synthesizer application with multiple oscillators,
    waveform selection, amplitude and frequency controls, DC offset, scalable time axis,
    trigger detection, volume control, octave keyboard interface, output device selection,
    key transposition, and the ability to save/load settings and play tunes from files.
    """

    # Configuration Constants
    SAMPLE_RATE = 44100  # Sample rate in Hz
    NUM_SINE_OSCILLATORS = 5  # Number of sine oscillators
    WAVEFORM_OPTIONS = ['Sine', 'Square', 'Sawtooth', 'Triangle']
    BUFFER_DURATION = 2.0  # Buffer duration in seconds for plotting and trigger detection
    PLOT_UPDATE_INTERVAL = 50  # Plot update interval in milliseconds
    TRIGGER_HOLDOFF_DEFAULT = 100.0  # Default trigger holdoff in milliseconds
    VOLUME_DEFAULT = 1.0  # Default volume (0.0 to 1.0)

    # Updated KEY_ORDER to include G3 up to G5 (two octaves)
    KEY_ORDER = [
        'G3', 'G#3/Ab3', 'A3', 'A#3/Bb3', 'B3',
        'C4', 'C#4/Db4', 'D4', 'D#4/Eb4', 'E4', 'F4', 'F#4/Gb4', 'G4',
        'G#4/Ab4', 'A4', 'A#4/Bb4', 'B4',
        'C5', 'C#5/Db5', 'D5', 'D#5/Eb5', 'E5', 'F5', 'F#5/Gb5', 'G5'
    ]

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.num_oscillators = self.NUM_SINE_OSCILLATORS  # Total oscillators
        self.amp_values = np.zeros(self.num_oscillators)  # Amplitudes for oscillators
        self.freq_values = np.zeros(self.num_oscillators)  # Base frequencies for oscillators
        self.phases = np.zeros(self.num_oscillators)  # Phase accumulators for oscillators
        self.waveform_types = ['Sine'] * self.num_oscillators  # Waveform types for oscillators
        self.dc_value = 0.0  # DC Offset
        self.volume = self.VOLUME_DEFAULT  # Volume (0.0 to 1.0)
        self.freq_multiplier_step = None  # Frequency Multiplier step (None corresponds to no key pressed)
        self.freq_multiplier = 0.0  # Frequency scaling factor
        self.cumulative_buffer = deque(maxlen=int(self.BUFFER_DURATION * self.SAMPLE_RATE))
        self.trigger_buffer = deque(maxlen=int((self.BUFFER_DURATION * 2) * self.SAMPLE_RATE))
        self.last_trigger_time = -np.inf  # To track holdoff
        self.stream = None  # Audio stream
        self.plotTimer = None  # Plot update timer
        self.currentKey = None  # Current key pressed
        self.output_device_indices = []  # List of output device indices

        # Cached Trigger Parameters for Thread Safety
        self.trigger_threshold = 0.0
        self.trigger_edge = 'Rising'

        # Tune Playback Variables
        self.tune_timer = QtCore.QTimer()
        self.tune_timer.timeout.connect(self.playNextNote)
        self.current_tune = []
        self.current_note_index = 0
        self.is_playing_tune = False
        self.is_paused = False  # Indicates if playback is paused

        # Recording Attributes
        self.is_recording = False
        self.recorded_notes = []
        self.recording_start_time = None
        self.current_recording_note = None
        self.quantization = Quantization.QUARTER  # Default quantization

        # Key Transposition Attributes
        self.key_shift_semitones = 0  # Number of semitones to shift
        self.key_shift_factor = 1.0  # Frequency scaling factor based on semitone shift

        self.initUI()
        self.initAudio()
        self.initPlotTimer()

    def initUI(self):
        """Initialize the User Interface."""
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.leftLayout = QtWidgets.QGridLayout()
        self.rightLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.addLayout(self.leftLayout)
        self.mainLayout.addLayout(self.rightLayout)
        self.initOscillatorControls()
        self.initDCControl()
        self.initTimeScaleControl()
        self.initTriggerControls()
        self.initRecordingControls()  # Initialize Recording Controls
        self.initOctaveKeyboard()
        self.initOutputDeviceSelection()
        self.initSaveLoadButtons()
        self.initWaveformPlot()
        self.initVolumeControl()
        self.initKeyTransposition()  # Initialize Key Transposition Controls
        self.setLayout(self.mainLayout)
        self.setWindowTitle('Fourier Synthesizer')
        self.updateParameters()

    def initKeyTransposition(self):
        """Initialize Key Transposition controls."""
        # Positioning below the existing controls with ample spacing
        trans_row = self.num_oscillators + 12  # Adjusted row number to prevent overlap

        # Label for Key Shift (updated color to black)
        trans_label = QtWidgets.QLabel('Key Shift (semitones)')
        trans_label.setStyleSheet("color: black; font-weight: bold;")  # Changed from white to black
        self.leftLayout.addWidget(trans_label, trans_row, 0)

        # SpinBox for Key Shift
        self.keyShiftSpinBox = QtWidgets.QSpinBox()
        self.keyShiftSpinBox.setRange(-24, 24)  # Allow shifting up or down by two octaves
        self.keyShiftSpinBox.setSingleStep(1)
        self.keyShiftSpinBox.setValue(0)
        self.keyShiftSpinBox.setToolTip("Shift the key up or down by semitones.")
        self.leftLayout.addWidget(self.keyShiftSpinBox, trans_row, 1)

        # Additional Label 'Key shift'
        key_shift_label = QtWidgets.QLabel('Key shift')
        key_shift_label.setStyleSheet("color: white; font-weight: bold;")
        self.leftLayout.addWidget(key_shift_label, trans_row, 2)

        # Connect SpinBox to updateTransposition without causing hangs
        self.keyShiftSpinBox.valueChanged.connect(self.updateTransposition)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(trans_row, 30)

    def updateTransposition(self, value):
        """
        Update the frequency scaling factor based on key transposition.

        Args:
            value (int): Number of semitones to shift (positive for up, negative for down).
        """
        with self.lock:
            self.key_shift_semitones = value
            self.key_shift_factor = 2 ** (self.key_shift_semitones / 12)
        # No need to call updateParameters here as key_shift_factor is used directly in audioCallback

    def initRecordingControls(self):
        """Initialize Recording controls: Start, Stop, Quantization selection."""
        # Adjusted row to prevent overlap with ample spacing
        recording_row = self.num_oscillators + 6  # Updated row number

        # Start Recording Button
        self.startRecordingButton = QtWidgets.QPushButton("Start Recording")
        self.startRecordingButton.setToolTip("Begin recording your melody.")
        self.startRecordingButton.clicked.connect(self.startRecording)
        self.leftLayout.addWidget(self.startRecordingButton, recording_row, 0)

        # Stop Recording Button
        self.stopRecordingButton = QtWidgets.QPushButton("Stop Recording")
        self.stopRecordingButton.setToolTip("End recording and prepare to save.")
        self.stopRecordingButton.clicked.connect(self.stopRecording)
        self.stopRecordingButton.setEnabled(False)  # Disabled initially
        self.leftLayout.addWidget(self.stopRecordingButton, recording_row, 1)

        # Quantization Selection
        quantization_label = QtWidgets.QLabel("Quantization")
        quantization_label.setStyleSheet("color: white; font-weight: bold;")
        self.leftLayout.addWidget(quantization_label, recording_row, 2)

        self.quantizationComboBox = QtWidgets.QComboBox()
        self.quantizationComboBox.addItems(["Whole Notes", "Half Notes", "Quarter Notes"])
        self.quantizationComboBox.setToolTip("Select the quantization level for recorded notes.")
        self.quantizationComboBox.currentIndexChanged.connect(self.changeQuantization)
        self.leftLayout.addWidget(self.quantizationComboBox, recording_row, 3)

        # Recording Status Label
        self.recordingStatusLabel = QtWidgets.QLabel("Status: Not Recording")
        self.recordingStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        self.leftLayout.addWidget(self.recordingStatusLabel, recording_row, 4)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(recording_row, 40)

    def changeQuantization(self, index):
        """Change the quantization level based on user selection."""
        if index == 0:
            self.quantization = Quantization.WHOLE
        elif index == 1:
            self.quantization = Quantization.HALF
        elif index == 2:
            self.quantization = Quantization.QUARTER

    def startRecording(self):
        """Start the recording process."""
        self.is_recording = True
        self.recorded_notes = []
        self.recording_start_time = time.time()
        self.recordingStatusLabel.setText("Status: Recording...")
        self.recordingStatusLabel.setStyleSheet("color: green; font-weight: bold;")
        self.startRecordingButton.setEnabled(False)
        self.stopRecordingButton.setEnabled(True)

    def stopRecording(self):
        """Stop the recording process and prompt to save."""
        self.is_recording = False
        self.recordingStatusLabel.setText("Status: Not Recording")
        self.recordingStatusLabel.setStyleSheet("color: red; font-weight: bold;")
        self.startRecordingButton.setEnabled(True)
        self.stopRecordingButton.setEnabled(False)

        if not self.recorded_notes:
            QtWidgets.QMessageBox.warning(self, "No Notes Recorded", "You haven't recorded any notes.")
            return

        # Prompt user to save the recording
        self.saveRecording()

    def initOscillatorControls(self):
        """Initialize oscillator amplitude, frequency, and waveform controls."""
        self.amplitudeControls = []
        self.frequencyControls = []
        self.waveformControls = []
        self.waveLabels = []

        for i in range(self.num_oscillators):
            wave_type = 'Sine'
            wave_num = i + 1
            color = 'green'  # Green color for sine waves

            label = QtWidgets.QLabel(f'{wave_type} Wave {wave_num}')
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            self.waveLabels.append(label)

            # Amplitude SpinBox (Adjusted Range)
            amp_spinbox = QtWidgets.QDoubleSpinBox()
            amp_spinbox.setRange(0.0, 1.0)  # Changed from -1.0 to 1.0 to remove negative amplitudes
            amp_spinbox.setSingleStep(0.0001)
            amp_spinbox.setDecimals(4)
            amp_spinbox.setValue(0.0)
            amp_spinbox.setToolTip("Adjust the amplitude of the oscillator.")
            self.amplitudeControls.append(amp_spinbox)

            # Frequency SpinBox
            freq_spinbox = QtWidgets.QDoubleSpinBox()
            freq_spinbox.setRange(20.0, self.SAMPLE_RATE / 2)
            freq_spinbox.setSingleStep(0.01)
            freq_spinbox.setDecimals(2)
            freq_spinbox.setValue(261.63)  # Default frequency set to C4
            freq_spinbox.setToolTip("Set the frequency of the oscillator in Hz.")
            self.frequencyControls.append(freq_spinbox)

            # Waveform ComboBox
            waveform_combo = QtWidgets.QComboBox()
            waveform_combo.addItems(self.WAVEFORM_OPTIONS)
            waveform_combo.setCurrentText('Sine')
            waveform_combo.setToolTip("Select the waveform type for the oscillator.")
            self.waveformControls.append(waveform_combo)

            # Add widgets to the grid layout
            row = i  # Each oscillator on a separate row
            self.leftLayout.addWidget(label, row, 0)
            self.leftLayout.addWidget(QtWidgets.QLabel('Amplitude'), row, 1)
            self.leftLayout.addWidget(amp_spinbox, row, 2)
            self.leftLayout.addWidget(QtWidgets.QLabel('Frequency (Hz)'), row, 3)
            self.leftLayout.addWidget(freq_spinbox, row, 4)
            self.leftLayout.addWidget(QtWidgets.QLabel('Waveform'), row, 5)
            self.leftLayout.addWidget(waveform_combo, row, 6)

            # Connect controls to the parameter update function
            amp_spinbox.valueChanged.connect(self.updateParameters)
            freq_spinbox.valueChanged.connect(self.updateParameters)
            waveform_combo.currentIndexChanged.connect(self.updateParameters)

            # Add spacing to prevent overlapping
            self.leftLayout.setRowMinimumHeight(row, 30)

    def initDCControl(self):
        """Initialize DC Offset control."""
        dc_row = self.num_oscillators + 1
        dc_label = QtWidgets.QLabel('DC Offset')
        dc_label.setStyleSheet("color: cyan; font-weight: bold;")
        self.leftLayout.addWidget(dc_label, dc_row, 0)

        self.dcSpinBox = QtWidgets.QDoubleSpinBox()
        self.dcSpinBox.setRange(-1.0, 1.0)
        self.dcSpinBox.setSingleStep(0.0001)
        self.dcSpinBox.setDecimals(4)
        self.dcSpinBox.setValue(0.0)
        self.dcSpinBox.setToolTip("Adjust the DC offset of the waveform.")
        self.leftLayout.addWidget(self.dcSpinBox, dc_row, 1)

        self.dcSpinBox.valueChanged.connect(self.updateParameters)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(dc_row, 30)

    def initTimeScaleControl(self):
        """Initialize Time Scale control."""
        time_scale_row = self.num_oscillators + 2
        time_scale_label = QtWidgets.QLabel('Time Scale (s)')
        time_scale_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.leftLayout.addWidget(time_scale_label, time_scale_row, 0)

        self.timeScaleSpinBox = QtWidgets.QDoubleSpinBox()
        self.timeScaleSpinBox.setRange(0.00002, 2.0)
        self.timeScaleSpinBox.setSingleStep(0.00001)
        self.timeScaleSpinBox.setDecimals(5)
        self.timeScaleSpinBox.setValue(0.05)
        self.timeScaleSpinBox.setToolTip("Set the duration of the waveform display in seconds.")
        self.leftLayout.addWidget(self.timeScaleSpinBox, time_scale_row, 1)

        self.timeScaleSpinBox.valueChanged.connect(self.updateParameters)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(time_scale_row, 30)

    def initTriggerControls(self):
        """Initialize Trigger controls."""
        trigger_row = self.num_oscillators + 3
        # Trigger Threshold
        trigger_threshold_label = QtWidgets.QLabel('Trigger Threshold')
        trigger_threshold_label.setStyleSheet("color: magenta; font-weight: bold;")
        self.leftLayout.addWidget(trigger_threshold_label, trigger_row, 0)

        self.triggerThresholdSpinBox = QtWidgets.QDoubleSpinBox()
        self.triggerThresholdSpinBox.setRange(-1.0, 1.0)
        self.triggerThresholdSpinBox.setSingleStep(0.01)
        self.triggerThresholdSpinBox.setDecimals(2)
        self.triggerThresholdSpinBox.setValue(0.0)
        self.triggerThresholdSpinBox.setToolTip("Set the amplitude threshold for trigger detection.")
        self.leftLayout.addWidget(self.triggerThresholdSpinBox, trigger_row, 1)

        # Trigger Edge
        trigger_edge_label = QtWidgets.QLabel('Trigger Edge')
        trigger_edge_label.setStyleSheet("color: magenta; font-weight: bold;")
        self.leftLayout.addWidget(trigger_edge_label, trigger_row, 2)

        self.triggerEdgeComboBox = QtWidgets.QComboBox()
        self.triggerEdgeComboBox.addItems(['Rising', 'Falling'])
        self.triggerEdgeComboBox.setCurrentText('Rising')
        self.triggerEdgeComboBox.setToolTip("Select the edge type for trigger detection.")
        self.leftLayout.addWidget(self.triggerEdgeComboBox, trigger_row, 3)

        # Connect Trigger controls to updateParameters
        self.triggerThresholdSpinBox.valueChanged.connect(self.updateParameters)
        self.triggerEdgeComboBox.currentIndexChanged.connect(self.updateParameters)

        # Trigger Holdoff
        holdoff_row = self.num_oscillators + 4
        holdoff_label = QtWidgets.QLabel('Trigger Holdoff (ms)')
        holdoff_label.setStyleSheet("color: orange; font-weight: bold;")
        self.leftLayout.addWidget(holdoff_label, holdoff_row, 0)

        self.triggerHoldoffSpinBox = QtWidgets.QDoubleSpinBox()
        self.triggerHoldoffSpinBox.setRange(0.0, 1000.0)
        self.triggerHoldoffSpinBox.setSingleStep(10.0)
        self.triggerHoldoffSpinBox.setDecimals(1)
        self.triggerHoldoffSpinBox.setValue(self.TRIGGER_HOLDOFF_DEFAULT)
        self.triggerHoldoffSpinBox.setToolTip("Set the holdoff time in milliseconds to prevent rapid triggering.")
        self.leftLayout.addWidget(self.triggerHoldoffSpinBox, holdoff_row, 1)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(holdoff_row, 30)

    def initVolumeControl(self):
        """Initialize Volume control as a vertical slider on the right side with adjusted layout."""
        volume_label = QtWidgets.QLabel('Volume')
        volume_label.setStyleSheet("color: blue; font-weight: bold;")
        volume_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rightLayout.addWidget(volume_label)

        # Create a vertical box layout for slider and percentage label
        volume_layout = QtWidgets.QVBoxLayout()
        self.volumeSlider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(int(self.volume * 100))
        self.volumeSlider.setTickInterval(10)
        self.volumeSlider.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.volumeSlider.setToolTip("Adjust the overall output volume.")
        # Shorten the slider height to be a 10th of its original length
        self.volumeSlider.setFixedHeight(100)  # Adjusted from default to 100 pixels

        self.volumeLabel = QtWidgets.QLabel(f'{int(self.volume * 100)}%')
        self.volumeLabel.setStyleSheet("color: blue; font-weight: bold;")
        self.volumeLabel.setAlignment(QtCore.Qt.AlignCenter)

        # Add slider and label to the vertical layout
        volume_layout.addWidget(self.volumeSlider)
        volume_layout.addWidget(self.volumeLabel)

        # Add the vertical layout to the rightLayout
        self.rightLayout.addLayout(volume_layout)

        # Connect Volume Slider to updateVolume
        self.volumeSlider.valueChanged.connect(self.updateVolume)

    def initOctaveKeyboard(self):
        """Initialize the Octave Keyboard without the label and extended to G3 and G5."""
        keyboard_row = self.num_oscillators + 5
        # Removed the Octave Keyboard label as per user request

        self.keyboardLayout = QtWidgets.QHBoxLayout()
        self.keyboardButtons = {}
        self.currentKey = None

        # Define the keys from G3 up to G5 (two octaves)
        keys = self.KEY_ORDER

        # Define which keys are black keys
        black_keys = [
            'G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
            'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5'
        ]

        for key in keys:
            btn = QtWidgets.QPushButton(key)
            if key in black_keys:
                btn.setStyleSheet("background-color: black; color: white;")
                btn.setFixedSize(30, 100)
            else:
                btn.setStyleSheet("background-color: white; color: black;")
                btn.setFixedSize(40, 150)
            # Connect both pressed and released signals
            btn.pressed.connect(self.onKeyPressed)
            btn.released.connect(self.onKeyReleased)
            self.keyboardLayout.addWidget(btn)
            self.keyboardButtons[key] = btn

        self.leftLayout.addLayout(self.keyboardLayout, keyboard_row, 1, 1, 6)

        # Label to display current key
        # Moved "Current Key" label to a new row to make room
        current_key_row = keyboard_row + 1
        self.currentKeyLabel = QtWidgets.QLabel(f'Current Key: None')
        self.currentKeyLabel.setStyleSheet("color: red; font-weight: bold;")
        self.leftLayout.addWidget(self.currentKeyLabel, current_key_row, 7)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(current_key_row, 30)

    def initOutputDeviceSelection(self):
        """Initialize Output Device Selection."""
        device_combo_row = self.num_oscillators + 7
        device_label = QtWidgets.QLabel('Output Device')
        device_label.setStyleSheet("color: white; font-weight: bold;")
        self.leftLayout.addWidget(device_label, device_combo_row, 0)

        self.deviceComboBox = QtWidgets.QComboBox()
        self.output_device_indices = []
        devices = self.getOutputDeviceNames()
        if not devices:
            self.deviceComboBox.addItem("No Output Devices Found")
            self.deviceComboBox.setEnabled(False)
        else:
            for idx, name in devices:
                self.output_device_indices.append(idx)
                self.deviceComboBox.addItem(name)
            self.deviceComboBox.setEnabled(True)
        if self.output_device_indices:
            self.device = self.output_device_indices[0]
        else:
            self.device = None  # Handle case with no output devices
        self.deviceComboBox.setToolTip("Select the audio output device.")
        self.deviceComboBox.currentIndexChanged.connect(self.changeDevice)

        self.leftLayout.addWidget(self.deviceComboBox, device_combo_row, 1, 1, 6)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(device_combo_row, 30)

    def initSaveLoadButtons(self):
        """Initialize Save and Load Buttons."""
        save_load_row = self.num_oscillators + 8
        # Save and Load All Settings
        save_button = QtWidgets.QPushButton("Save All Settings")
        load_button = QtWidgets.QPushButton("Load All Settings")
        save_button.setToolTip("Save all synthesizer settings to a JSON file.")
        load_button.setToolTip("Load synthesizer settings from a JSON file.")
        self.leftLayout.addWidget(save_button, save_load_row, 0)
        self.leftLayout.addWidget(load_button, save_load_row, 1)

        # Connect buttons to their respective functions
        save_button.clicked.connect(self.saveAllSettings)
        load_button.clicked.connect(self.loadAllSettings)

        # Save and Load Oscillator Settings
        osc_save_load_row = save_load_row + 1
        save_osc_button = QtWidgets.QPushButton("Save Oscillator Settings")
        load_osc_button = QtWidgets.QPushButton("Load Oscillator Settings")
        save_osc_button.setToolTip("Save only oscillator amplitudes, frequencies, and waveforms to a JSON file.")
        load_osc_button.setToolTip("Load oscillator amplitudes, frequencies, and waveforms from a JSON file.")
        self.leftLayout.addWidget(save_osc_button, osc_save_load_row, 0)
        self.leftLayout.addWidget(load_osc_button, osc_save_load_row, 1)

        # Connect oscillator buttons to their respective functions
        save_osc_button.clicked.connect(self.saveOscillatorSettings)
        load_osc_button.clicked.connect(self.loadOscillatorSettings)

        # Add Load Tune, Play Tune, Stop Tune, and Pause/Resume Tune buttons
        tune_buttons_row = osc_save_load_row + 1
        load_tune_button = QtWidgets.QPushButton("Load Tune")
        play_tune_button = QtWidgets.QPushButton("Play Tune")
        stop_tune_button = QtWidgets.QPushButton("Stop Tune")
        self.pauseResumeTuneButton = QtWidgets.QPushButton("Pause Tune")  # New Pause/Resume Button
        self.pauseResumeTuneButton.setToolTip("Pause or resume the currently playing tune.")
        self.pauseResumeTuneButton.setEnabled(False)  # Disabled initially

        load_tune_button.setToolTip("Load a tune from a JSON file.")
        play_tune_button.setToolTip("Play the loaded tune automatically.")
        stop_tune_button.setToolTip("Stop the currently playing tune.")
        self.pauseResumeTuneButton.setToolTip("Pause or resume the currently playing tune.")

        self.leftLayout.addWidget(load_tune_button, tune_buttons_row, 0)
        self.leftLayout.addWidget(play_tune_button, tune_buttons_row, 1)
        self.leftLayout.addWidget(stop_tune_button, tune_buttons_row, 2)
        self.leftLayout.addWidget(self.pauseResumeTuneButton, tune_buttons_row, 3)

        # Connect tune buttons to their respective functions
        load_tune_button.clicked.connect(self.loadTune)
        play_tune_button.clicked.connect(self.playTune)
        stop_tune_button.clicked.connect(self.stopTune)
        self.pauseResumeTuneButton.clicked.connect(self.pauseResumeTune)  # Connect Pause/Resume

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(tune_buttons_row, 30)

    def initWaveformPlot(self):
        """Initialize the Waveform Visualization Plot."""
        plot_row = self.num_oscillators + 9  # Updated row number to prevent overlap
        self.plotWidget = pg.PlotWidget(background='k', foreground='g')
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)
        self.plotCurve = self.plotWidget.plot([], [], pen=pg.mkPen(color='lime', width=2))
        self.plotWidget.setYRange(-1, 1)
        self.plotWidget.setXRange(0, self.timeScaleSpinBox.value())
        self.leftLayout.addWidget(self.plotWidget, plot_row, 0, 1, 8)

        # Add Trigger Line Indicator
        self.triggerLine = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('g', width=2))
        self.plotWidget.addItem(self.triggerLine)

        # Add spacing to prevent overlapping
        self.leftLayout.setRowMinimumHeight(plot_row, 200)

    def getOutputDeviceNames(self):
        """
        Retrieve a list of available output devices.

        Returns:
            List of tuples containing device index and name.
        """
        try:
            devices = sd.query_devices()
            output_devices = []
            for idx, device in enumerate(devices):
                if device['max_output_channels'] > 0:
                    name = device['name']
                    if isinstance(name, tuple):
                        # Join tuple elements into a single string
                        name = "/".join(name)
                    elif not isinstance(name, str):
                        # Convert to string if it's not already
                        name = str(name)
                    output_devices.append((idx, name))
            if not output_devices:
                QtWidgets.QMessageBox.warning(self, "No Output Devices", "No audio output devices found.")
            return output_devices
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to query audio devices:\n{e}")
            return []

    def changeDevice(self, index):
        """
        Change the audio output device based on user selection.

        Args:
            index (int): Index of the selected device in the combo box.
        """
        if not self.output_device_indices:
            QtWidgets.QMessageBox.warning(self, "No Output Device", "No output devices are available.")
            return
        try:
            selected_device = self.output_device_indices[index]
            if selected_device != self.device:
                self.device = selected_device
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
                self.initAudio()
                # Removed the popup message to avoid unnecessary notifications
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to change output device:\n{e}")

    def initAudio(self):
        """Initialize the audio stream."""
        if self.device is None:
            QtWidgets.QMessageBox.warning(self, "No Output Device", "No output device selected.")
            return
        with self.lock:
            self.phases = np.zeros(self.num_oscillators)  # Reset phases

        try:
            self.stream = sd.OutputStream(device=self.device,
                                          samplerate=self.SAMPLE_RATE,
                                          channels=1,
                                          callback=self.audioCallback)
            self.stream.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to initialize audio stream:\n{e}")

    def initPlotTimer(self):
        """Initialize the plot update timer."""
        self.plotTimer = QtCore.QTimer()
        self.plotTimer.setInterval(self.PLOT_UPDATE_INTERVAL)
        self.plotTimer.timeout.connect(self.updatePlot)
        self.plotTimer.start()

    def audioCallback(self, outdata, frames, time_info, status):
        """
        Callback function for audio stream.

        Args:
            outdata (np.ndarray): Output buffer to fill.
            frames (int): Number of frames to generate.
            time_info (dict): Time information.
            status (CallbackFlags): Status flags.
        """
        if status:
            print(status)
        out = np.zeros(frames)

        # Thread-safe access to parameters using cached values
        with self.lock:
            amp_values = self.amp_values.copy()
            freq_values = self.freq_values.copy()
            phases = self.phases.copy()
            waveforms = self.waveform_types.copy()
            dc = self.dc_value
            trigger_threshold = self.trigger_threshold
            trigger_edge = self.trigger_edge
            plot_duration = self.timeScaleSpinBox.value()
            volume = self.volume
            freq_multiplier = self.freq_multiplier
            key_shift_factor = self.key_shift_factor

        if freq_multiplier == 0.0 and dc == 0.0:
            # Output silence if no frequency multiplier and DC offset is zero
            out = np.zeros(frames)
        else:
            for i in range(self.num_oscillators):
                amp = amp_values[i]
                freq = freq_values[i] * freq_multiplier * key_shift_factor
                phase_increment = 2 * np.pi * freq / self.SAMPLE_RATE
                phase_array = phases[i] + phase_increment * np.arange(frames)

                # Generate waveform based on selected type
                waveform = waveforms[i]
                if waveform == 'Sine':
                    out += amp * np.sin(phase_array)
                elif waveform == 'Square':
                    out += amp * np.sign(np.sin(phase_array))
                elif waveform == 'Sawtooth':
                    # Improved Sawtooth Waveform Generation
                    out += amp * (2 * (phase_array / (2 * np.pi) - np.floor(phase_array / (2 * np.pi) + 0.5)))
                elif waveform == 'Triangle':
                    out += amp * (2 * np.abs(
                        2 * (phase_array / (2 * np.pi) - np.floor(phase_array / (2 * np.pi) + 0.5))) - 1)

                # Update phase for continuity
                phases[i] += phase_increment * frames
                phases[i] %= 2 * np.pi  # Keep phase within [0, 2π]

            # Add DC offset
            out += dc

            # Apply volume control
            out *= volume

            # Normalize the output to prevent clipping
            max_amp = np.max(np.abs(out)) + abs(dc)
            if max_amp > 1.0:
                out /= max_amp  # Normalize considering DC offset

        # Write to audio buffer
        outdata[:, 0] = out

        # Update phases in a thread-safe manner
        with self.lock:
            self.phases = phases

            if freq_multiplier != 0.0 or dc != 0.0:
                # Accumulate data for plotting
                self.cumulative_buffer.extend(out)
                # Accumulate data for trigger detection
                self.trigger_buffer.extend(out)
            else:
                # If silent, clear buffers to prevent unnecessary accumulation
                self.cumulative_buffer.clear()
                self.trigger_buffer.clear()

    def detectTrigger(self, buffer, threshold, edge):
        """
        Detect the trigger event in the buffer.

        Args:
            buffer (deque): The audio samples buffer.
            threshold (float): The amplitude threshold for triggering.
            edge (str): 'Rising' or 'Falling'.

        Returns:
            int or None: The index where the trigger occurs, or None if not found.
        """
        buffer = np.array(buffer)
        if edge == 'Rising':
            # Look for a transition from below to above the threshold
            crossings = np.where((buffer[:-1] < threshold) & (buffer[1:] >= threshold))[0]
        else:
            # Look for a transition from above to below the threshold
            crossings = np.where((buffer[:-1] > threshold) & (buffer[1:] <= threshold))[0]

        if crossings.size > 0:
            # Return the last trigger point found for stability
            return crossings[-1] + 1  # +1 to get the index after crossing
        else:
            return None

    def updatePlot(self):
        """Update the waveform visualization plot."""
        with self.lock:
            trigger_buffer = list(self.trigger_buffer)
            duration = self.timeScaleSpinBox.value()
            trigger_threshold = self.trigger_threshold
            trigger_edge = self.trigger_edge
            holdoff_time = self.triggerHoldoffSpinBox.value()

        if len(trigger_buffer) == 0:
            return  # Nothing to plot

        # Detect trigger in the trigger buffer
        trigger_index = self.detectTrigger(trigger_buffer, trigger_threshold, trigger_edge)

        if trigger_index is not None:
            # Get the current time in milliseconds
            current_time = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch()
            # Check holdoff to prevent multiple rapid triggers
            if (current_time - self.last_trigger_time) >= holdoff_time:
                # Align the plot starting from the trigger point
                aligned_buffer = trigger_buffer[trigger_index:]
                self.last_trigger_time = current_time  # Update last trigger time

                # Position the trigger line at the start
                self.triggerLine.setPos(0)
            else:
                # Ignore triggers within holdoff time
                aligned_buffer = trigger_buffer[-int(duration * self.SAMPLE_RATE):]
                self.triggerLine.setPos(0)  # Reset trigger line position
        else:
            # If no trigger found, use the most recent samples
            aligned_buffer = trigger_buffer[-int(duration * self.SAMPLE_RATE):]
            self.triggerLine.setPos(0)  # Reset trigger line position

        # Adjust the buffer to fit the plot duration
        if len(aligned_buffer) > int(duration * self.SAMPLE_RATE):
            aligned_buffer = aligned_buffer[-int(duration * self.SAMPLE_RATE):]

        t = np.linspace(0, len(aligned_buffer) / self.SAMPLE_RATE, len(aligned_buffer), endpoint=False)
        waveform = np.array(aligned_buffer)

        # Update the plot data
        self.plotCurve.setData(t, waveform)

        # Adjust Y-axis range based on maximum amplitude including DC
        max_amp = np.max(np.abs(waveform))
        if max_amp == 0:
            max_amp = 1  # Avoid division by zero
        self.plotWidget.setYRange(-max_amp, max_amp)
        self.plotWidget.setXRange(0, duration)  # X-axis starts at 0

    def updateParameters(self, value=None):
        """
        Update amplitude, frequency, waveform types, DC, Time Scale, Trigger settings,
        Frequency Multiplier, and Key Shift in a thread-safe manner.
        """
        with self.lock:
            for i in range(self.num_oscillators):
                self.amp_values[i] = self.amplitudeControls[i].value()
                self.freq_values[i] = self.frequencyControls[i].value()
                self.waveform_types[i] = self.waveformControls[i].currentText()
            self.dc_value = self.dcSpinBox.value()  # Update DC offset
            # Update cached trigger parameters
            self.trigger_threshold = self.triggerThresholdSpinBox.value()
            self.trigger_edge = self.triggerEdgeComboBox.currentText()
            # freq_multiplier and key_shift_factor are updated elsewhere

    def onKeyPressed(self):
        """
        Handle keyboard button press events to adjust frequency multiplier and record the note.
        """
        sender = self.sender()
        key = sender.text()
        if key in self.KEY_ORDER:
            n = self.KEY_ORDER.index(key)  # Number of semitones above G3
            scaling_factor = 2 ** (n / 12)
            with self.lock:
                self.freq_multiplier_step = n
                self.freq_multiplier = scaling_factor
                self.currentKey = key
                # Reset phases to ensure each note starts fresh
                self.phases = np.zeros(self.num_oscillators)
            # Update the label to reflect the current key
            self.currentKeyLabel.setText(f'Current Key: {self.currentKey}')
            # Visually indicate the selected key
            for k, btn in self.keyboardButtons.items():
                if k == key:
                    btn.setStyleSheet("background-color: blue; color: white;")
                elif k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                           'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                    btn.setStyleSheet("background-color: black; color: white;")
                else:
                    btn.setStyleSheet("background-color: white; color: black;")

            # Recording Logic
            if self.is_recording:
                current_time = time.time() - self.recording_start_time
                self.current_recording_note = {
                    "name": key,
                    "start_time": current_time,
                    "end_time": None  # To be filled on release
                }

    def onKeyReleased(self):
        """
        Handle keyboard button release events to reset frequency multiplier and record the note duration.
        """
        sender = self.sender()
        key = sender.text()
        if key in self.KEY_ORDER:
            with self.lock:
                self.freq_multiplier_step = None
                self.freq_multiplier = 0.0
                self.currentKey = None
                # Reset phases to silence
                self.phases = np.zeros(self.num_oscillators)
            self.currentKeyLabel.setText(f'Current Key: None')
            # Visually reset the released key and highlight None
            for k, btn in self.keyboardButtons.items():
                if k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                         'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                    btn.setStyleSheet("background-color: black; color: white;")
                else:
                    btn.setStyleSheet("background-color: white; color: black;")

            # Recording Logic
            if self.is_recording and self.current_recording_note:
                current_time = time.time() - self.recording_start_time
                self.current_recording_note["end_time"] = current_time
                self.recorded_notes.append(self.current_recording_note)
                self.current_recording_note = None

    def updateVolume(self, value):
        """
        Update the volume based on the slider value.

        Args:
            value (int): Volume percentage (0 to 100).
        """
        with self.lock:
            self.volume = value / 100.0  # Convert percentage to 0.0 - 1.0
        self.volumeLabel.setText(f"{value}%")

    def saveAllSettings(self):
        """
        Save all synthesizer settings to a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save All Settings", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            settings = {
                'amplitudes': self.amp_values.tolist(),
                'frequencies': self.freq_values.tolist(),
                'waveforms': self.waveform_types,
                'dc_offset': self.dc_value,
                'time_scale': self.timeScaleSpinBox.value(),
                'trigger_threshold': self.trigger_threshold,
                'trigger_edge': self.trigger_edge,
                'trigger_holdoff': self.triggerHoldoffSpinBox.value(),
                'volume': self.volume,  # Already a float
                'frequency_multiplier_step': self.freq_multiplier_step,  # Save multiplier step
                'key_shift_semitones': self.key_shift_semitones  # Save key shift
            }
            try:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=4)
                QtWidgets.QMessageBox.information(self, "Success", f"All settings saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")

    def loadAllSettings(self):
        """
        Load all synthesizer settings from a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load All Settings", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                # Validate JSON keys
                required_keys = ['amplitudes', 'frequencies', 'waveforms', 'dc_offset', 'time_scale',
                                 'trigger_threshold', 'trigger_edge', 'trigger_holdoff',
                                 'volume', 'frequency_multiplier_step', 'key_shift_semitones']
                for key in required_keys:
                    if key not in settings:
                        raise ValueError(f"Missing key in JSON: {key}")

                # Validate lengths
                amplitudes = settings.get('amplitudes', [0.0] * self.num_oscillators)
                frequencies = settings.get('frequencies', [261.63] * self.num_oscillators)
                waveforms = settings.get('waveforms', ['Sine'] * self.num_oscillators)
                if len(amplitudes) != self.num_oscillators or \
                        len(frequencies) != self.num_oscillators or \
                        len(waveforms) != self.num_oscillators:
                    raise ValueError("Amplitude, frequency, and waveform lists must match the number of oscillators.")

                # Block signals to prevent updateParameters from being called multiple times
                for spinbox in self.amplitudeControls + self.frequencyControls + [
                    self.dcSpinBox, self.timeScaleSpinBox,
                    self.triggerThresholdSpinBox, self.triggerHoldoffSpinBox,
                    self.keyShiftSpinBox
                ]:
                    spinbox.blockSignals(True)
                for combo in self.waveformControls:
                    combo.blockSignals(True)
                self.triggerEdgeComboBox.blockSignals(True)
                self.volumeSlider.blockSignals(True)

                # Apply settings
                for i in range(self.num_oscillators):
                    self.amplitudeControls[i].setValue(amplitudes[i])
                    self.frequencyControls[i].setValue(frequencies[i])
                    waveform = waveforms[i]
                    if waveform in self.WAVEFORM_OPTIONS:
                        self.waveformControls[i].setCurrentText(waveform)
                    else:
                        self.waveformControls[i].setCurrentText('Sine')  # Default if invalid
                self.dcSpinBox.setValue(settings.get('dc_offset', 0.0))
                self.timeScaleSpinBox.setValue(settings.get('time_scale', 0.05))
                self.triggerThresholdSpinBox.setValue(settings.get('trigger_threshold', 0.0))
                edge = settings.get('trigger_edge', 'Rising')
                index = self.triggerEdgeComboBox.findText(edge)
                if index != -1:
                    self.triggerEdgeComboBox.setCurrentIndex(index)
                self.triggerHoldoffSpinBox.setValue(settings.get('trigger_holdoff', self.TRIGGER_HOLDOFF_DEFAULT))
                volume = settings.get('volume', self.VOLUME_DEFAULT)
                self.volumeSlider.setValue(int(volume * 100))
                self.volumeLabel.setText(f"{int(volume * 100)}%")
                freq_multiplier_step = settings.get('frequency_multiplier_step', None)
                self.freq_multiplier_step = freq_multiplier_step
                key_shift_semitones = settings.get('key_shift_semitones', 0)
                self.keyShiftSpinBox.setValue(key_shift_semitones)

                # Apply key transposition
                with self.lock:
                    self.key_shift_semitones = key_shift_semitones
                    self.key_shift_factor = 2 ** (self.key_shift_semitones / 12)

                if self.freq_multiplier_step is not None and 0 <= self.freq_multiplier_step < len(self.KEY_ORDER):
                    self.freq_multiplier = 2 ** (self.freq_multiplier_step / 12)
                    selected_key = self.KEY_ORDER[self.freq_multiplier_step]
                    self.currentKey = selected_key
                    self.currentKeyLabel.setText(f'Current Key: {self.currentKey}')
                    # Highlight selected key
                    for k, btn in self.keyboardButtons.items():
                        if k == selected_key:
                            btn.setStyleSheet("background-color: blue; color: white;")
                        elif k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                                   'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                            btn.setStyleSheet("background-color: black; color: white;")
                        else:
                            btn.setStyleSheet("background-color: white; color: black;")
                else:
                    # If step is None or out of range, reset to no key pressed
                    self.freq_multiplier_step = None
                    self.freq_multiplier = 0.0
                    self.currentKey = None
                    self.currentKeyLabel.setText(f'Current Key: None')
                    # Reset key styles
                    for k, btn in self.keyboardButtons.items():
                        if k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                                 'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                            btn.setStyleSheet("background-color: black; color: white;")
                        else:
                            btn.setStyleSheet("background-color: white; color: black;")

                # Unblock signals
                for spinbox in self.amplitudeControls + self.frequencyControls + [
                    self.dcSpinBox, self.timeScaleSpinBox,
                    self.triggerThresholdSpinBox, self.triggerHoldoffSpinBox,
                    self.keyShiftSpinBox
                ]:
                    spinbox.blockSignals(False)
                for combo in self.waveformControls:
                    combo.blockSignals(False)
                self.triggerEdgeComboBox.blockSignals(False)
                self.volumeSlider.blockSignals(False)

                # Update internal parameters
                self.updateParameters()

                QtWidgets.QMessageBox.information(self, "Success", f"All settings loaded from {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load settings:\n{e}")

    def saveOscillatorSettings(self):
        """
        Save only oscillator settings (amplitudes, frequencies, and waveforms) to a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Oscillator Settings", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            osc_settings = {
                'amplitudes': self.amp_values.tolist(),
                'frequencies': self.freq_values.tolist(),
                'waveforms': self.waveform_types
            }
            try:
                with open(filename, 'w') as f:
                    json.dump(osc_settings, f, indent=4)
                QtWidgets.QMessageBox.information(self, "Success", f"Oscillator settings saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save oscillator settings:\n{e}")

    def loadOscillatorSettings(self):
        """
        Load only oscillator settings (amplitudes, frequencies, and waveforms) from a JSON file.
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Oscillator Settings", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    osc_settings = json.load(f)
                # Validate JSON keys
                required_keys = ['amplitudes', 'frequencies', 'waveforms']
                for key in required_keys:
                    if key not in osc_settings:
                        raise ValueError(f"Missing key in JSON: {key}")

                # Validate lengths
                amplitudes = osc_settings.get('amplitudes', [0.0] * self.num_oscillators)
                frequencies = osc_settings.get('frequencies', [261.63] * self.num_oscillators)
                waveforms = osc_settings.get('waveforms', ['Sine'] * self.num_oscillators)
                if len(amplitudes) != self.num_oscillators or \
                        len(frequencies) != self.num_oscillators or \
                        len(waveforms) != self.num_oscillators:
                    raise ValueError("Amplitude, frequency, and waveform lists must match the number of oscillators.")

                # Block signals to prevent updateParameters from being called multiple times
                for spinbox in self.amplitudeControls + self.frequencyControls:
                    spinbox.blockSignals(True)
                for combo in self.waveformControls:
                    combo.blockSignals(True)

                # Apply oscillator settings
                for i in range(self.num_oscillators):
                    self.amplitudeControls[i].setValue(amplitudes[i])
                    self.frequencyControls[i].setValue(frequencies[i])
                    waveform = waveforms[i]
                    if waveform in self.WAVEFORM_OPTIONS:
                        self.waveformControls[i].setCurrentText(waveform)
                    else:
                        self.waveformControls[i].setCurrentText('Sine')  # Default if invalid

                # Unblock signals
                for spinbox in self.amplitudeControls + self.frequencyControls:
                    spinbox.blockSignals(False)
                for combo in self.waveformControls:
                    combo.blockSignals(False)

                # Update internal parameters
                self.updateParameters()

                QtWidgets.QMessageBox.information(self, "Success", f"Oscillator settings loaded from {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load oscillator settings:\n{e}")

    def loadTune(self):
        """Load a tune from a JSON file."""
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Tune", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            try:
                with open(filename, 'r') as f:
                    tune = json.load(f)
                notes = tune.get('notes', [])
                if not notes:
                    raise ValueError("No notes found in the tune file.")
                self.current_tune = notes
                self.current_note_index = 0
                self.is_playing_tune = False  # Don't start playback immediately
                QtWidgets.QMessageBox.information(self, "Tune Loaded",
                                                  f"Loaded tune: {tune.get('tune_name', 'Unnamed')}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load tune:\n{e}")

    def playTune(self):
        """Play the loaded tune automatically."""
        if not self.current_tune:
            QtWidgets.QMessageBox.warning(self, "No Tune Loaded", "Please load a tune before attempting to play.")
            return
        if self.is_playing_tune:
            QtWidgets.QMessageBox.warning(self, "Already Playing", "A tune is already being played.")
            return
        self.is_playing_tune = True
        self.is_paused = False
        self.current_note_index = 0
        self.pauseResumeTuneButton.setText("Pause Tune")
        self.pauseResumeTuneButton.setEnabled(True)
        self.playNextNote()

    def pauseResumeTune(self):
        """Toggle between pausing and resuming the tune playback."""
        if not self.is_playing_tune:
            return  # No tune is playing

        if not self.is_paused:
            # Pause the playback
            self.is_paused = True
            self.tune_timer.stop()
            self.pauseResumeTuneButton.setText("Resume Tune")
        else:
            # Resume the playback
            self.is_paused = False
            self.pauseResumeTuneButton.setText("Pause Tune")
            self.playNextNote()

    def playNextNote(self):
        """Play the next note in the current tune."""
        if not self.is_playing_tune or self.current_note_index >= len(self.current_tune):
            self.is_playing_tune = False
            self.pauseResumeTuneButton.setEnabled(False)
            return

        note = self.current_tune[self.current_note_index]
        note_name = note.get('name')
        duration = note.get('duration', 1.0)  # Default to 1 second if duration not specified

        if note_name not in self.KEY_ORDER:
            # Skip invalid notes
            self.current_note_index += 1
            self.playNextNote()
            return

        # Simulate key press
        scaling_factor = 2 ** (self.KEY_ORDER.index(note_name) / 12)
        with self.lock:
            self.freq_multiplier_step = self.KEY_ORDER.index(note_name)
            self.freq_multiplier = scaling_factor
            self.currentKey = note_name
            # Reset phases to ensure each note starts fresh
            self.phases = np.zeros(self.num_oscillators)
        self.currentKeyLabel.setText(f'Current Key: {self.currentKey}')

        # Visually indicate the selected key if within range after transposition
        transposed_key = self.getTransposedKey(note_name)
        if transposed_key and transposed_key in self.KEY_ORDER:
            display_key = transposed_key
            for k, btn in self.keyboardButtons.items():
                if k == display_key:
                    btn.setStyleSheet("background-color: blue; color: white;")
                elif k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                           'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                    btn.setStyleSheet("background-color: black; color: white;")
                else:
                    btn.setStyleSheet("background-color: white; color: black;")
        else:
            # Do not change color if transposed key is out of range
            for k, btn in self.keyboardButtons.items():
                if k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                         'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                    btn.setStyleSheet("background-color: black; color: white;")
                else:
                    btn.setStyleSheet("background-color: white; color: black;")

        # Schedule the next note after the current one finishes
        self.tune_timer.singleShot(int(duration * 1000), self.endCurrentNote)

    def getTransposedKey(self, key):
        """
        Get the transposed key based on the current key shift.

        Args:
            key (str): Original key name.

        Returns:
            str or None: Transposed key name if within range, else None.
        """
        original_index = self.KEY_ORDER.index(key)
        transposed_index = original_index + self.key_shift_semitones
        if 0 <= transposed_index < len(self.KEY_ORDER):
            return self.KEY_ORDER[transposed_index]
        else:
            return None  # Out of range

    def endCurrentNote(self):
        """End the current note and proceed to the next one."""
        # Reset frequency multiplier
        with self.lock:
            self.freq_multiplier_step = None
            self.freq_multiplier = 0.0
            self.currentKey = None
            # Reset phases to ensure next note starts fresh
            self.phases = np.zeros(self.num_oscillators)
        self.currentKeyLabel.setText(f'Current Key: None')

        # Reset key styles
        for k, btn in self.keyboardButtons.items():
            if k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                     'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                btn.setStyleSheet("background-color: black; color: white;")
            else:
                btn.setStyleSheet("background-color: white; color: black;")

        self.current_note_index += 1
        if not self.is_paused:
            self.playNextNote()

    def stopTune(self):
        """Stop the currently playing tune."""
        if self.is_playing_tune:
            self.tune_timer.stop()
            self.is_playing_tune = False
            self.is_paused = False
            self.current_note_index = 0
            self.pauseResumeTuneButton.setText("Pause Tune")
            self.pauseResumeTuneButton.setEnabled(False)
            # Reset frequency multiplier
            with self.lock:
                self.freq_multiplier_step = None
                self.freq_multiplier = 0.0
                self.currentKey = None
                # Reset phases to silence
                self.phases = np.zeros(self.num_oscillators)
            self.currentKeyLabel.setText(f'Current Key: None')
            # Reset key styles
            for k, btn in self.keyboardButtons.items():
                if k in ['G#3/Ab3', 'A#3/Bb3', 'C#4/Db4', 'D#4/Eb4', 'F#4/Gb4',
                         'G#4/Ab4', 'A#4/Bb4', 'C#5/Db5', 'D#5/Eb5', 'F#5/Gb5']:
                    btn.setStyleSheet("background-color: black; color: white;")
                else:
                    btn.setStyleSheet("background-color: white; color: black;")

    def quantize_notes(self):
        """
        Quantize the recorded notes to the nearest quantization level.

        Returns:
            List of quantized note dictionaries with 'name' and 'duration'.
        """
        quantized_notes = []
        for note in self.recorded_notes:
            if note["end_time"] is None:
                continue  # Skip if end_time wasn't recorded
            duration = note["end_time"] - note["start_time"]
            quantized_duration = self.quantization.value

            # Find the nearest quantization level
            if duration < (self.quantization.value / 2):
                quantized_duration = self.quantization.value / 2
            elif duration > self.quantization.value * 1.5:
                quantized_duration = self.quantization.value * 1.5

            quantized_notes.append({
                "name": note["name"],
                "duration": quantized_duration
            })
        return quantized_notes

    def saveRecording(self):
        """Save the recorded and quantized notes as a JSON file."""
        quantized_notes = self.quantize_notes()

        # Prompt user to enter a tune name
        tune_name, ok = QtWidgets.QInputDialog.getText(self, "Tune Name", "Enter a name for your tune:")
        if not ok or not tune_name.strip():
            tune_name = "Recorded Tune"

        # Create the JSON structure
        tune = {
            "tune_name": tune_name,
            "notes": quantized_notes
        }

        # Prompt user to choose a file location
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Recorded Tune", "", "JSON Files (*.json)",
                                                            options=options)
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(tune, f, indent=4)
                QtWidgets.QMessageBox.information(self, "Success", f"Tune saved successfully to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save tune:\n{e}")

    def closeEvent(self, event):
        """
        Handle the window close event to ensure resources are properly released.

        Args:
            event (QCloseEvent): The close event.
        """
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
            if self.plotTimer is not None and self.plotTimer.isActive():
                self.plotTimer.stop()
            if self.tune_timer.isActive():
                self.tune_timer.stop()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        event.accept()


def main():
    """Main function to run the application."""
    app = QtWidgets.QApplication(sys.argv)
    synth = FourierSynthesizer()
    synth.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
