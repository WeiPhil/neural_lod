#!/bin/python3
import imageio
import numpy as np
import sys

transmission_layer = imageio.imread(sys.argv[1])
transmission_layer = np.mean(transmission_layer, axis=(0, 1))
transmission_layer = np.floor(transmission_layer) / 255.0

if not transmission_layer[0] > 0 and not transmission_layer[2] > 0:
    #print("Ignoring transmission layer %s" % sys.argv[2])
    sys.exit(1)
print("Writing transmission layer %s" % sys.argv[2])

ior = transmission_layer[1] * 2.0 - 1.0
if ior > 1.0:
    ior = 1.0 / np.sqrt(1.0 - ior)
else:
    ior = np.sqrt(ior + 1.0)

with open(sys.argv[2], 'w') as f:
    f.write('%f %f %f' % (transmission_layer[2], ior, transmission_layer[0]))
