#! /bin/bash

#javacommon_jar=/home/fresheed/AndroidStudioProjects/ActionLogger/javacommon/build/libs/javacommon.jar


if [ -n "$1" ]; then
    filter="^$1_.*_log$"
else
    filter="_log$"
    rm -rf parsed_logs
    mkdir parsed_logs
fi

echo "filter:" $filter

for log in $(ls -p raw_logs | grep $filter); do 
    echo "Parsing $log"
    groovy -cp javacommon.jar decode_logs.groovy raw_logs/$log > parsed_logs/$log.txt
done


