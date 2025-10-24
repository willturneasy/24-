#!/bin/bash
used=`free -m | awk 'NR==2' | awk '{print $3}'`
free=`free -m | awk 'NR==2' | awk '{print $4}'`

echo "==========================="
echo "内存使用情况 | [使用了:${used}MB][剩余:${free}MB]"

if [ $free -le 2000 ] ; then
                sync && echo 1 > /proc/sys/vm/drop_caches
                echo "清理Cached成功"
else
                echo "不需要清理"
fi
exit