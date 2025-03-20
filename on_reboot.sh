#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pibd/.Xauthority
xhost +SI:Localuser:pi-bd

cd /home/pi-bd/Documents/photobooth_musee_bd
source venv/bin/activate
python3 main.py >> /home/pi-bd/log_cron.txt 2>&1
