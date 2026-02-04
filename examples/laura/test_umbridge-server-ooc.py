# run this script as:       python3 test_umbridge-server-ooc.py http://localhost:4242  

import argparse
import umbridge
import numpy as np

parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
parser.add_argument('url',metavar='url',type=str,
                    help='the URL on which the model is running, for example http://localhost:4242')
args = parser.parse_args()
print(f"Connecting to host URL {args.url}")

# Set up a model by connecting to URL
model = umbridge.HTTPModel(args.url,"forward")

#test get input method
output = model.get_input_sizes()
print(output)

#test get output method
output = model.get_output_sizes()
print(output)


#test model output
param = [[ 250.,  600.,   650.,   250.,  1.0*10**(-6),  1.0*10**(-6),   1.0*10**(-7),   1.0*10**(-4),  50.0 ]]
output = model(param)
print(output)

