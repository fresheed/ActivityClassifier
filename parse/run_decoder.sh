#! /bin/bash

#javacommon_jar=/home/fresheed/AndroidStudioProjects/ActionLogger/javacommon/build/libs/javacommon.jar

rm -rf parsed_logs
mkdir parsed_logs

for log in $(ls -p raw_logs | grep '_log$'); do 
    echo "Parsing $log"
    groovy -cp javacommon.jar decode_logs.groovy raw_logs/$log > parsed_logs/$log.txt
done


