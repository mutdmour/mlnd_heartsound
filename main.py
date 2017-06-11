import wave

FNAME = './heartbeat-sounds/set_a/normal__201101070538.wav'

f = wave.open(FNAME)

# frames will hold the bytestring representing all the audio frames
frames = f.readframes(-1)
#print(frames)
#print(frames[:20].decode("utf-8") )

import struct
samples = struct.unpack('h'*f.getnframes(), frames)
print(samples[:10])


framerate = f.getframerate()
t = [float(i)/framerate for i in range(len(samples))]
print(t[:10])


from pylab import *
plot(t, samples)
