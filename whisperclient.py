import collections
import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import socket
import threading
import logging
import json

SampleRate = 44100          # Stream device recording frequency
BlockSize = 30              # Block size in milliseconds
Threshold = 0.1             # Minimum volume threshold to activate listening
Vocal_Range = [50, 1000]    # Frequency range to detect sounds that could be speech
EndBlocks = 100             # Number of blocks of silence to wait before sending to Whisper
Internal = 50               # Number of blocks to wait before sending to Whisper (without silence)
Debug = True                # Enables debug GUI elements
Max_Plot = 500              # The maximum number of points on the frequency plot.

data_y = collections.deque([0.0, 0.0],maxlen=Max_Plot)
data_x = collections.deque([0.0, 0.0],maxlen=Max_Plot)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

class StreamHandler:
    def __init__(self):
        self.running = True
        self.padding = 0
        self.interval = 0
        self.prevblock = self.buffer = np.zeros((0,1))
        self.fileready = False
        self.newsegment = False
        self.forcerecord = False

    def receive_response(self, sock):
        while True:
            # Wait for a response from the Whisper server
            data = sock.recv(1024)

            # If there is no more data to receive, break out of the loop
            if not data:
                break

            # Print the response from the Whisper server to the GUI
            logging.debug("Data received from whisper server! newsegment: " + str(self.newsegment) + " fileready: " + str(self.fileready))
            json_value = data.decode()
            message = json.loads(json_value)
            new_value = message['text']
            if message['endsegment']:
                self.newsegment = False
                dpg.set_value("transcripion", '')
                old_value = dpg.get_value("transcripion-previous") + '\n'
                dpg.set_value("transcripion-previous", old_value + new_value)
                new_value = ''
            dpg.set_value("transcripion", new_value)

            

        # Close the socket
        sock.close()
        
    def callback(self, indata, frames, time, status):
        if not any(indata) and not self.forcerecord:
            return
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames

        # Update GUI
        data_x.append(1 + data_x[-1])
        data_y.append(freq if freq < Vocal_Range[1] else Vocal_Range[1])
        dpg.set_value('series_tag', [list(data_x), list(data_y)])
        dpg.fit_axis_data('x_axis')
        dpg.fit_axis_data('y_axis')

        # Check if sound in threshold
        if (np.sqrt(np.mean(indata**2)) > Threshold and Vocal_Range[0] <= freq <= Vocal_Range[1]) or self.forcerecord:           
            # If this is not the first block in the sequence
            if self.padding < 1: 
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
            if (self.interval == 0):
                self.interval = Internal
            dpg.set_value("status","Recording...")
        else:
            self.padding -= 1
            if Debug:
                dpg.set_value("interval", "interval: " + str(self.interval))
                dpg.set_value("shape", "buffer.shape: " + str(self.buffer.shape[0]))
            # Continue recording voice if not enough silence has passed
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            # if enough silence has passed, write to file.
            elif self.padding < 1 < self.buffer.shape[0] > SampleRate: 
                self.fileready = True
                self.newsegment = True
                write('dictate.wav', SampleRate, self.buffer) 
                self.buffer = np.zeros((0,1))
                dpg.set_value("status","Silence")
                logging.debug("File sent to Whisper after prolonged silence")
            # if recording not long enough, reset buffer.
            elif self.padding < 1 < self.buffer.shape[0] < SampleRate: 
                self.buffer = np.zeros((0,1))
                print("\033[2K\033[0G", end='', flush=True)
                dpg.set_value("status","Silence")
                logging.debug("Buffer reset. Insifficient size")
            else:
                self.prevblock = indata.copy() 
                dpg.set_value("status","Silence")
        # Periodically save file and send to Whisper 
        self.interval = 0 if self.interval == 0 else self.interval -1 
        if not self.fileready and self.interval == 0 and self.buffer.shape[0] > SampleRate:
            self.fileready = True
            self.interval = Internal
            write('dictate.wav', SampleRate, self.buffer) 
            logging.debug("File periodically sent to WHhisper after set amount of time")

    def process(self, sock):
        dpg.render_dearpygui_frame()
        if self.fileready:
            message = {'translate' : False, 'english' : True, "endsegment" : self.newsegment}
            message = json.dumps(message)
            sock.sendall(message.encode())
            self.fileready = False

    def listen(self):
        dpg.add_button(label="Force", callback=print("Hello"), parent=main_window)
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the Whisper server is listening
        try:
            server_address = ('localhost', 9999)
            sock.connect(server_address)
        except(ConnectionRefusedError):
            print("Could not connect to socket server. Make sure whisperserver.py is running.")
            return

        # Create a thread to receive responses from file 2
        receive_thread = threading.Thread(target=self.receive_response, args=(sock,))
        receive_thread.start()

        print("\033[32mListening.. \033[37m(Ctrl+C to Quit)\033[0m")
        with sd.InputStream(channels=1, callback=self.callback, blocksize=int(SampleRate * BlockSize / 1000), samplerate=SampleRate):
            while dpg.is_dearpygui_running():
                self.process(sock)

# GUI Code
dpg.create_context()

def clear_text():
    dpg.set_value("transcripion-previous", '')

with dpg.window(label="Live Whisper", tag="Primary Window") as main_window:
    # Frequency plot
    with dpg.plot(label="Microphone Frequency", height=200, width=400):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag='x_axis')
        dpg.add_plot_axis(dpg.mvYAxis, label="Frequency", tag="y_axis")
        dpg.add_line_series(list(data_x), list(data_y), parent="y_axis", tag="series_tag")
    dpg.add_combo(label="Task", items=['Translation', 'Transcription'], default_value='Transcription', width=120)
    dpg.add_text("Silence", tag="status")
    dpg.add_text("", tag="transcripion-previous", wrap=1000)
    dpg.add_text("", tag="transcripion", wrap=1000)
    if Debug:
        dpg.add_text("interval: ", tag="interval")
        dpg.add_text("buffer.shape : ", tag="shape")
        dpg.add_text("Sample rate; " + str(SampleRate), tag="sample-rate")
    dpg.add_button(label="Clear Text", callback=clear_text)

dpg.create_viewport()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

def main():
    handler = StreamHandler()
    handler.listen()
        
if __name__ == '__main__':
    main()  