import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import socket
import threading

SampleRate = 44100  # Stream device recording frequency
BlockSize = 30      # Block size in milliseconds
Threshold = 0.1     # Minimum volume threshold to activate listening
Vocals = [50, 1000] # Frequency range to detect sounds that could be speech
EndBlocks = 100     # Number of blocks of silence to wait before sending to Whisper
Internal = 50       # Number of blocks to wait before sending to Whisper (without silence)
Debug = True


class StreamHandler:
    def __init__(self):
        self.running = True
        self.padding = 0
        self.interval = 0
        self.prevblock = self.buffer = np.zeros((0,1))
        self.fileready = False
        self.newsegment = False

    def receive_response(self, sock):
        while True:
            # Wait for a response from the whisper server
            data = sock.recv(1024)

            # If there is no more data to receive, break out of the loop
            if not data:
                break

            # Print the response from the whisper server to the GUI
            new_value = data.decode()
            if self.newsegment:
                self.newsegment = False
                dpg.set_value("transcripion", '')
                old_value = dpg.get_value("transcripion-previous") + '\n'
                dpg.set_value("transcripion-previous", old_value + new_value)
                new_value = ''
            dpg.set_value("transcripion", new_value)
            self.fileready = False
            

        # Close the socket
        sock.close()
        
    def callback(self, indata, frames, time, status):
        if not any(indata):
            return
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames

        # Update GUI
        dpg.set_value("frequency", "Frequency: " + str(round(freq,2)))
        old_value = dpg.get_value("progress-bar")
        if (old_value > freq/Vocals[1]):
            new_value = max(old_value-0.01,0)
        elif (old_value < freq/Vocals[1]):
            new_value = min(old_value+0.01,1.0)
        else:
            new_value = old_value
        dpg.set_value("progress-bar", new_value)

        # Check if sound in threshold
        if np.sqrt(np.mean(indata**2)) > Threshold and Vocals[0] <= freq <= Vocals[1]:           
            # If this is not the first block in the sequence
            if self.padding < 1: 
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
            if (self.interval == 0):
                self.interval = Internal
        else:
            self.padding -= 1
            self.interval = 0 if self.interval == 0 else self.interval -1 
            if Debug:
                dpg.set_value("interval", "interval: " + str(self.interval))
                dpg.set_value("shape", "buffer.shape: " + str(self.buffer.shape[0]))
            # Periodically save file and send to Whisper 
            if not self.fileready and self.interval == 0 and self.buffer.shape[0] > SampleRate:
                self.fileready = True
                write('dictate.wav', SampleRate, self.buffer) 
                print("1")
            # Continue recording voice if not enough silence has passed
            if self.padding > 1:
                dpg.set_value("status","Recording...")
                self.buffer = np.concatenate((self.buffer, indata))
            # if enough silence has passed, write to file.
            elif self.padding < 1 < self.buffer.shape[0] > SampleRate: 
                self.fileready = True
                self.newsegment = True
                write('dictate.wav', SampleRate, self.buffer) 
                self.buffer = np.zeros((0,1))
                dpg.set_value("status","Silence")
                print("2")
            # if recording not long enough, reset buffer.
            elif self.padding < 1 < self.buffer.shape[0] < SampleRate: 
                self.buffer = np.zeros((0,1))
                print("\033[2K\033[0G", end='', flush=True)
                dpg.set_value("status","Silence")
                print("3")
            else:
                self.prevblock = indata.copy() 
                dpg.set_value("status","Silence")

    def process(self, sock):
        dpg.render_dearpygui_frame()
        if self.fileready:
            #print("\n\033[90mTranscribing..\033[0m")
            message = 'send_hello_world'
            sock.sendall(message.encode())

    def listen(self):
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where thw whisper server is listening
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

with dpg.window(label="Live Whisper", tag="Primary Window"):
    textControl = dpg.add_text("Clicks: 0", tag="frequency")
    dpg.add_progress_bar(tag="progress-bar")
    dpg.set_value("progress-bar", 0.3)
    dpg.add_text("Silence", tag="status")
    dpg.add_text("", tag="transcripion-previous", wrap=1000)
    dpg.add_text("", tag="transcripion", wrap=1000)
    if Debug:
        dpg.add_text("interval: ", tag="interval")
        dpg.add_text("buffer.shape : ", tag="shape")
        dpg.add_text("Sample rate; " + str(SampleRate), tag="sample-rate")
    dpg.add_button(label="Clear Text")

dpg.create_viewport()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

def main():
    handler = StreamHandler()
    handler.listen()
        
if __name__ == '__main__':
    main()  