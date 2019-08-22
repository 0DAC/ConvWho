#!/bin/bash

echo "Beginning split in this directory"
echo "Are you sure you want to proceed? (Y/N)"
read answer

if [ $answer = Y ]; then
	swap_status=0

	for filename in *.jpg.chip.jpg; do
		echo "Evaluating $filename"
		if [ $((swap_status%2)) = 1 ]; then
			echo "Moving $filename to $1"
			mv $filename $1
			rm $filename
		fi
		swap_status=$((swap_status+1))
	done
	echo "Directory Split Complete"
else
	echo "Aborted"
fi
