#!/bin/sh
# Find a UART for the ECG data link on the ZCU104 and, if asked, free the console UART.
#
#   sh uart_setup.sh                 # report only: serial devices, which one is the console
#   sh uart_setup.sh free ttyPS0     # stop the login getty + kernel console chatter on ttyPS0
#
# Freeing the console UART means you lose the serial shell on it: make sure
# SSH over Ethernet works first (ip addr; ssh root@<board-ip>). A reboot restores it.

set -u

echo "== Serial devices =="
ls -l /dev/ttyPS* /dev/ttyUL* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "(none found)"

echo
echo "== Kernel console (from /proc/cmdline) =="
tr ' ' '\n' < /proc/cmdline | grep '^console=' || echo "(no console= argument)"

echo
echo "== getty services on serial ports =="
(systemctl list-units --all 2>/dev/null | grep -i 'getty@tty') || \
    (ps 2>/dev/null | grep -i '[g]etty') || echo "(could not list getty processes)"

echo
echo "== UART controllers in device tree =="
for node in /proc/device-tree/axi/serial@* /proc/device-tree/amba/serial@*; do
    [ -e "$node" ] || continue
    status="okay"
    [ -f "$node/status" ] && status=$(tr -d '\0' < "$node/status")
    echo "$(basename "$node"): status=$status"
done

if [ "${1:-}" = "free" ]; then
    dev="${2:?usage: sh uart_setup.sh free ttyPS0}"
    echo
    echo "== Freeing /dev/$dev for data =="
    systemctl stop "serial-getty@$dev.service" 2>/dev/null || true
    pkill -f "getty.*$dev" 2>/dev/null || true
    # Stop kernel log messages from being printed on the console UART mid-frame.
    dmesg -n 1 2>/dev/null || echo 1 > /proc/sys/kernel/printk
    echo "Done. Kernel messages still go to dmesg; reboot to restore the serial console."
    echo "If frames still arrive corrupted, check: fuser /dev/$dev"
fi

echo
echo "Next: python3 uart_server.py --xmodel ecg_denoiser.xmodel --serial /dev/<tty> --baud 115200 --power"
