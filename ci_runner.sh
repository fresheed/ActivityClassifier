#! /bin/sh

python3 -m parse.ci_load_logs --token $dropbox_token --local_dir raw_logs
python3 -m unittest discover
