#!/bin/bash
echo "Connecting to Jevois cli"

# list the device and copy its identifier
deviceId="$(ls /dev/tty.usbmodem*)"
screen $deviceId 115200
