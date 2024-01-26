"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

def getAngles(*args):
  print(args[:])

# 0 skeloton number of angles
# -4: the four angles



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="10.20.17.122", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=8100, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = Dispatcher()
  dispatcher.map("/tracker", getAngles)


  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()