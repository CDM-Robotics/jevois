#!/bin/bash
keepAlive=true

echo "Making jevois disk accessible"
echo usbsd > /dev/cu.usbmodem*

#echo -n "To eject the disk, type Exit and press [ENTER]: "

while true; do
	echo -n "Enter a command and press [Enter]: "
	read user_cmd
	if [ $user_cmd == 'exit' ]; then
		echo "Ejecting the disk...."
		diskutil eject /dev/disk2
	elif [ $user_cmd == 'help' ]; then
		echo "COMMANDS"
		echo "exit -> eject the disk"
	fi
done