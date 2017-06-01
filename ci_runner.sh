#! /bin/sh

python3 -m parse.ci_load_logs \
    --token $dropbox_token \
    --local_dir $local_logs_dir \
    --archive_name $archive_name
echo "Files loaded to $local_logs_dir"

python3 -m unittest discover

python3 -m classification.experiments.run_experiment \
    --algorithm ci \
    --log_dir $local_logs_dir \
    --classes pushups5_ walk50_ sits10_ typing_

#--classes pushups5_ walk50_ sits10_ typing_
#--classes bigramNA bigramST bigramTO
#--classes letterA letterB letterV

