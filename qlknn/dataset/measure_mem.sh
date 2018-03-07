#!/bin/bash
# 3 columns
echo Just input
/usr/bin/time -v python memtest.py '' 'True' 'join' 'False' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo 3 columns
echo -n "old    "
/usr/bin/time -v python memtest.py '' 'True' 'join' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "fast   "
/usr/bin/time -v python memtest.py '_sep' 'True' 'join' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "bare   "
/usr/bin/time -v python memtest.py '_sep' 'False' 'bare' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "join   "
/usr/bin/time -v python memtest.py '_sep' 'False' 'join' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "concat "
/usr/bin/time -v python memtest.py '_sep' 'False' 'concat' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "merge  "
/usr/bin/time -v python memtest.py '_sep' 'False' 'merge' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "assign "
/usr/bin/time -v python memtest.py '_sep' 'False' 'assign' 'efeITG_GB', 'efiITG_GB', 'pfeITG_GB' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'

echo full
echo -n "old    "
/usr/bin/time -v python memtest.py '' 'True' 'join' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "fast   "
/usr/bin/time -v python memtest.py '_sep' 'True' 'join' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "bare   "
/usr/bin/time -v python memtest.py '_sep' 'False' 'bare' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "join   "
/usr/bin/time -v python memtest.py '_sep' 'False' 'join' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "concat "
/usr/bin/time -v python memtest.py '_sep' 'False' 'concat' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "merge  "
/usr/bin/time -v python memtest.py '_sep' 'False' 'merge' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
echo -n "assign "
/usr/bin/time -v python memtest.py '_sep' 'False' 'assign' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
# Super slow/memory intensive
#echo update
#/usr/bin/time -v python memtest.py '_sep' 'False' 'update' 2>&1 >/dev/null | grep 'Maximum resident' | cut -d ' ' -f 6 | awk '{print $1/1024/1024" GiB"}'
