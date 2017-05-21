#! /bin/sh

local_logs_dir=raw_logs
python3 -m parse.ci_load_logs --token $dropbox_token --local_dir $local_logs_dir
echo "Files loaded to $local_logs_dir"
python3 -m unittest discover
