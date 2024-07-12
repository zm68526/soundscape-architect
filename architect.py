import sys
import threading;
import os
import pyaudio
import numpy as np
import random
import wave
from pydub import AudioSegment
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sounddevice as sd
import scipy.io.wavfile as wav

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
OUTPUT_FILENAME = "recorded_audio.wav"

# Global variables
recording = False
frames = []
lastRecordingFilepath = ""
writing = False
playing = False

class AudioPlayer(QThread):
    def __init__(self, filename, extension):
        super().__init__()
        self.filename = filename
        self.playing = False
        self.volume = 1.0
        self.muted = False
        self.pan = 0.0  # 0.0 is center, -1.0 is full left, 1.0 is full right
        self.audio_source = AudioSegment.from_file(self.filename) # the complete audio
        self.audio = self.audio_source # the trimmed audio
        self.duration_in_seconds = self.audio.duration_seconds
        self.original_duration = self.duration_in_seconds
        self.trimLeft = -1
        self.trimRight = -1
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.position = 0
        self.removed = False
        self.looping = True
        self.extension = extension
        self.min_interval = -1
        self.max_interval = -1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.handle_timer_finish)

    def update_trim(self):
        if self.trimLeft > -1: # placeholder for when the audio should be untrimmed
            self.audio = self.audio_source[self.trimLeft:self.trimRight]
            self.duration_in_seconds = self.audio.duration_seconds
        else:
            self.audio = self.audio_source
            self.duration_in_seconds = self.audio.duration_seconds

    def run(self):
        # Opens the stream
        self.stream = self.p.open(format=self.p.get_format_from_width(self.audio.sample_width),
                                  channels=self.audio.channels,
                                  rate=self.audio.frame_rate,
                                  output=True,
                                  stream_callback=self.callback)
        self.stream.start_stream()
        while not self.removed and self.stream.is_active():
            self.sleep(1)

    # Runs while stream is open
    def callback(self, in_data, frame_count, time_info, status):
        # If not playing - returns blank data
        if not self.playing:
            self.position = 0
            return (np.zeros(frame_count * self.audio.channels).tobytes(), pyaudio.paContinue)
        
        # Calculates the chunk size and retrieves the next chunk of audio data
        bytes_per_frame = self.audio.sample_width * self.audio.channels
        chunk_size = frame_count * bytes_per_frame
        data = self.audio.raw_data[self.position:self.position + chunk_size]
        self.position += chunk_size

        # If we've reached the end of the audio, it loops back to the beginning
        if len(data) < chunk_size:
            if self.looping:
                self.position = 0
                remaining = chunk_size - len(data)
                data += self.audio.raw_data[:remaining]
            else:
                self.position = 0
                remaining = chunk_size - len(data)
                # data += self.audio.raw_data[:remaining]
                data += b'\x00' * remaining # Pad with silence
                self.playing = False
                if self.min_interval > -1:
                    # self.timer.start(self.seconds_to_milliseconds(random.uniform(self.min_interval, self.max_interval)))
                    randomHolder = random.uniform(self.min_interval, self.max_interval)
                    self.timer_thread = threading.Timer(randomHolder, self.handle_timer_finish)
                    self.timer_thread.start()

        # Convert to numpy array for volume and pan adjustment
        np_data = np.frombuffer(data, dtype=np.int16)
        np_data = np_data.reshape(-1, 2)  # Reshape to stereo
        np_data = np_data.astype(np.float32)
        
        # Apply volume
        np_data *= self.volume

        # Apply panning
        left_factor = np.sqrt(2) / 2.0 * (np.cos(self.pan) - np.sin(self.pan))
        right_factor = np.sqrt(2) / 2.0 * (np.cos(self.pan) + np.sin(self.pan))
        np_data[:, 0] *= left_factor
        np_data[:, 1] *= right_factor

        # Converts the processed numpy array back to bytes
        data = np_data.astype(np.int16).tobytes()

        # If muted, return silent data but still advance pointer
        if self.muted:
            return (np.zeros(len(data)).tobytes(), pyaudio.paContinue)
        
        return (data, pyaudio.paContinue)

    def toggle_play(self):
        if hasattr(self, 'timer_thread') and self.timer_thread.is_alive(): # prevents lingering timer threads
            self.timer_thread.cancel()
        self.position = 0 if self.playing else self.position
        self.playing = not self.playing

    def toggle_mute(self):
        self.muted = not self.muted

    def toggle_loop(self):
        self.looping = not self.looping

    def set_volume(self, value):
        self.volume = value / 100.0

    def set_pan(self, value):
        # Convert slider value (-100 to 100) to radians (-pi/4 to pi/4) for dB
        self.pan = value / 100.0 * np.pi / 4

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
    
    def handle_timer_finish(self):
        global playing
        if playing:
            self.playing = True

class Sound(QWidget):
    removed = pyqtSignal(object)
    valueFailure = pyqtSignal(object)
    valueSuccess = pyqtSignal(object)
    trimmingValueFailure = pyqtSignal(object)
    trimmingValueSuccess = pyqtSignal(object)

    # name - the name of the sound as it shows up in the list
    # filepath - the path to the file
    # extension - the extension (wav, mp3, etc)
    def __init__(self, name, filepath, extension, parent=None):
        super().__init__(parent)
        self.name = name
        self.audio_player = AudioPlayer(filepath, extension)
        self.audio_player.start()
        self.initUI(name)

    def initUI(self, name):
        layout = QHBoxLayout()

        # Label showing file's name
        self.name_label = QLabel(name)
        layout.addWidget(self.name_label)

        # Play button for individual components, used for debugging
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        # layout.addWidget(self.play_button)

        # Button for muting during playback
        self.mute_button = QPushButton("Mute")
        self.mute_button.clicked.connect(self.toggle_mute)
        layout.addWidget(self.mute_button)

        # Slider for adjusting volume
        self.volume_slider = QSlider(Qt.Vertical)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.change_volume)
        layout.addWidget(self.volume_slider)

        # Slider for adjusting pan
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setRange(-100,100)
        self.pan_slider.setValue(0)
        self.pan_slider.valueChanged.connect(self.change_pan)
        layout.addWidget(self.pan_slider)

        # Button to toggle between looping and one-shot
        self.looping_button = QPushButton('Looping')
        self.looping_button.clicked.connect(self.toggle_loop)
        layout.addWidget(self.looping_button)

        # One-shot repetition info
        self.one_shot_label_1 = QLabel("When one-shot, repeat every")
        layout.addWidget(self.one_shot_label_1)
        self.text_box1 = QLineEdit()
        layout.addWidget(self.text_box1)
        self.one_shot_label_2 = QLabel("to")
        layout.addWidget(self.one_shot_label_2)
        self.text_box2 = QLineEdit()
        layout.addWidget(self.text_box2)
        self.one_shot_label_3 = QLabel("seconds")
        layout.addWidget(self.one_shot_label_3)
        self.update_button = QPushButton('Update Interval')
        self.update_button.clicked.connect(self.update_interval)
        layout.addWidget(self.update_button)

        # Trim info
        self.trimming_label_1 = QLabel("Trim:")
        layout.addWidget(self.trimming_label_1)
        self.text_box3 = QLineEdit()
        layout.addWidget(self.text_box3)
        self.trimming_label_2 = QLabel("to")
        layout.addWidget(self.trimming_label_2)
        self.text_box4 = QLineEdit()
        layout.addWidget(self.text_box4)
        self.trimming_label_3 = QLabel("seconds")
        layout.addWidget(self.trimming_label_3)
        self.update_trimming_button = QPushButton('Trim')
        self.update_trimming_button.clicked.connect(self.update_trim)
        layout.addWidget(self.update_trimming_button)

        # Button to remove sound
        self.remove_button = QPushButton('Remove')
        self.remove_button.clicked.connect(self.remove_self)
        layout.addWidget(self.remove_button)

        self.setLayout(layout)

    def update_trim(self):
        if (self.text_box3.text() == '' and self.text_box4.text() == ''):
            self.audio_player.trimLeft = -1
            self.audio_player.trimRight = -1
            self.audio_player.update_trim()
            self.trimmingValueSuccess.emit(self)
            return
        try:
            min_interval = float(self.text_box3.text())
            max_interval = float(self.text_box4.text())

            if (min_interval > max_interval
                or max_interval > self.audio_player.original_duration
                or min_interval > self.audio_player.original_duration):
                self.trimmingValueFailure.emit(self)
            else:
                self.audio_player.trimLeft = self.seconds_to_milliseconds(min_interval)
                self.audio_player.trimRight = self.seconds_to_milliseconds(max_interval)
                self.trimmingValueSuccess.emit(self)
                self.audio_player.update_trim()
        except ValueError:
            self.audio_player.min_interval = -1
            self.audio_player.max_interval = -1
            self.trimmingValueFailure.emit(self)

    def seconds_to_milliseconds(self, seconds):
        return int(round(seconds, 3)*1000)

    def update_interval(self):
        if (self.text_box1.text() == '' and self.text_box2.text() == ''):
            self.audio_player.min_interval = -1
            self.audio_player.max_interval = -1
            self.valueSuccess.emit(self)
            return
        try:
            min_interval = float(self.text_box1.text())
            max_interval = float(self.text_box2.text())

            if min_interval > max_interval:
                self.valueFailure.emit(self)
            else:
                self.audio_player.min_interval = min_interval
                self.audio_player.max_interval = max_interval
                self.valueSuccess.emit(self)
        except ValueError:
            self.audio_player.min_interval = -1
            self.audio_player.max_interval = -1
            self.valueFailure.emit(self)

    def toggle_play(self):
        self.audio_player.toggle_play()
        self.play_button.setText("Pause" if self.audio_player.playing else "Play")

    def toggle_mute(self):
        self.audio_player.toggle_mute()
        self.mute_button.setText("Unmute" if self.audio_player.muted else "Mute")

    def toggle_loop(self):
        self.audio_player.toggle_loop()
        self.looping_button.setText("Looping" if self.audio_player.looping else "One-Shot")

    def remove_self(self):
        self.valueSuccess.emit(self)
        self.audio_player.removed = True
        self.audio_player.stop()
        self.removed.emit(self)
        self.setParent(None)
        self.deleteLater()

    def change_pan(self, value):
        self.audio_player.set_pan(value)
    
    def change_volume(self, value):
        self.audio_player.set_volume(value)
    
    def closeEvent(self, event):
        # self.audio_player.stop()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.pyaudio = pyaudio.PyAudio()
        self.count_label = QLabel('Current Sounds: 0')
        self.sounds = [] # List to keep track of sounds
        self.valueErrors = [] # Files with interval errors
        self.trimmingValueErrors = [] # Files with trimming errors
        self.file_path = ""
        self.initUI()

    def update_count_display(self):
        self.count_label.setText(f'Current Sounds: {len(self.sounds)}') # Updates label to show correct number of sounds

    def initUI(self):
        self.setWindowTitle("Soundscape Architect")
        self.setGeometry(100, 100, 1600, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.add_button = QPushButton('Add Sound')
        self.add_button.clicked.connect(self.add_sound)
        self.main_layout.addWidget(self.count_label)
        self.main_layout.addWidget(self.add_button)

        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.main_layout.addWidget(self.play_button)

        # Mute all button
        self.mute_button = QPushButton("Mute All")
        self.mute_button.clicked.connect(self.mute_all)
        self.main_layout.addWidget(self.mute_button)

        # Record button
        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.handle_recording_thread)
        self.main_layout.addWidget(self.record_button)

        # Export button - currently unused
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.handle_exporting_thread)
        # self.main_layout.addWidget(self.export_button)

        # Label for warning about interval errors
        self.warning_label = QLabel('')
        self.main_layout.addWidget(self.warning_label)

        # Label for warning about trimming errors
        self.trimming_warning_label = QLabel('')
        self.main_layout.addWidget(self.trimming_warning_label)
    
    def toggle_play(self):
        global playing

        if self.add_button.isEnabled():
            self.add_button.setDisabled(True)
        else:
            self.add_button.setDisabled(False)
        
        if self.play_button.text() == "Play": # If currently paused
            playing = True
            for x in self.sounds:
                x.toggle_play()
        else: # If currently playing
            playing = False
            for x in self.sounds:
                if x.audio_player.playing: # Only toggle play if the audio is playing to avoid accidentally triggering inactive oneshots
                    x.toggle_play()
                #if x.audio_player.timer.isActive():
                #    x.timer.stop()
        for x in self.sounds:
            x.looping_button.setEnabled(not x.looping_button.isEnabled())
            x.update_button.setEnabled(not x.update_button.isEnabled())
            x.update_trimming_button.setEnabled(not x.update_trimming_button.isEnabled())
            x.remove_button.setEnabled(not x.remove_button.isEnabled())
        self.play_button.setText("Pause" if self.play_button.text() == "Play" else "Play")

    def closeEvent(self, event):
        # self.audio_player.stop()
        event.accept()

    def add_sound (self):
        global lastRecordingFilepath

        file_path = ""
        if (lastRecordingFilepath == ""):
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Sound File", "", "Audio Files (*.wav *.mp3 *.ogg *.aac *.m4a)")
        else:
            file_path = lastRecordingFilepath
            lastRecordingFilepath = ""
        file_name = os.path.basename(file_path)
        name, extension = os.path.splitext(file_name)
        extension = extension[1:]

        if file_path:
            new_sound = Sound(name, file_path, extension)
            new_sound.removed.connect(self.remove_sound)
            new_sound.valueFailure.connect(self.handle_value_error)
            new_sound.valueSuccess.connect(self.handle_value_success)
            new_sound.trimmingValueFailure.connect(self.handle_trimming_value_error)
            new_sound.trimmingValueSuccess.connect(self.handle_trimming_value_success)
            self.main_layout.addWidget(new_sound)
            self.sounds.append(new_sound)
            self.update_count_display()

    def handle_trimming_value_error(self, sound):
        if not sound in self.trimmingValueErrors:
            self.trimmingValueErrors.append(sound)
        self.update_trimming_value_error()
    
    def handle_trimming_value_success(self, sound):
        if sound in self.trimmingValueErrors:
            self.trimmingValueErrors.remove(sound)
        self.update_trimming_value_error()

    def handle_value_success(self, sound):
        if sound in self.valueErrors:
            self.valueErrors.remove(sound)
        self.update_value_error()

    def handle_value_error(self, sound):
        if not sound in self.valueErrors:
            self.valueErrors.append(sound)
        self.update_value_error()
    
    def update_trimming_value_error(self):
        if (len(self.trimmingValueErrors) == 0):
            self.trimming_warning_label.setText("")
        elif (len(self.trimmingValueErrors) == 1):
            self.trimming_warning_label.setText("There was an error parsing the trim settings for " + self.trimmingValueErrors[0].name + ". Values must be entered as floats, the max interval must be greater than the min interval, and both values must be longer than the total length of the audio.")
        else:
            newText = "There were errors parsing the trim settings for the following files: "
            for x in self.trimmingValueErrors:
                newText = newText + x.name + ", "
            newText = newText[:-2]
            newText = newText + ". Values must be entered as floats, the max interval must be greater than the min interval, and both values must be shorter than the total length of the audio."
            self.trimming_warning_label.setText(newText)
        
    def update_value_error(self):
        if (len(self.valueErrors) == 0):
            self.warning_label.setText("")
        elif (len(self.valueErrors) == 1):
            self.warning_label.setText("There was an error parsing the interval for " + self.valueErrors[0].name + ". Values must be entered as floats and the max interval must be greater than the min interval.")
        else:
            newText = "There were errors parsing the interval for the following files: "
            for x in self.valueErrors:
                newText = newText + x.name + ", "
            newText = newText[:-2]
            newText = newText + ". Values must be entered as floats and the max interval must be greater than the min interval."
            self.warning_label.setText(newText)

    def remove_sound(self, sound):
        if (sound in self.sounds):
            self.sounds.remove(sound)
        self.update_count_display()

    def mute_all(self):
        for x in self.sounds:
            if not x.audio_player.muted:
                x.toggle_mute()

    def handle_recording_thread(self):
        global recording, frames, lastRecordingFilepath, writing
        if self.record_button.text() == "Record": # Not currently recording
            # threading.Thread(target=self.record).start()
            self.file_path, _ = QFileDialog.getSaveFileName(window, "Recorded Audio", "", "WAV Files (*.wav)")
            if not self.file_path: return
            recording = True
            frames = []
            self.record_button.setText("Stop Recording")
            self.add_button.setDisabled(True)
            self.mute_button.setDisabled(True)
            self.play_button.setDisabled(True)
            if self.play_button.text() == "Pause":
                self.toggle_play()
                self.add_button.setDisabled(True)
            threading.Thread(target=self.record_audio).start()
        elif self.record_button.text() == "Stop Recording":
            recording = False
            writing = True
            self.record_button.setText("Record")
            lastRecordingFilepath = self.file_path
            self.add_button.setDisabled(False)
            self.mute_button.setDisabled(False)
            self.play_button.setDisabled(False)
            while writing:
                1 + 1 # Wait until the file has been written before trying to add the sound
            self.add_sound()
    
    def record_audio(self):
        global recording, frames, writing

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        

        while recording:
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        writing = False
    
    def handle_exporting_thread(self):
        self.toggle_play()
        duration = 10
        print("Recording system")
        recording = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS)
        sd.wait()
        normalized = np.int16(recording * 32767)
        file_path, _ = QFileDialog.getSaveFileName(self, "Exported Audio", "", "WAV Files (*.wav)")
        wav.write(file_path, RATE, normalized)
        print("Recording saved")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())