#! /bin/sh

python3 -m parse.ci_load_logs \
    --token $dropbox_token \
    --local_dir $local_logs_dir \
    --archives_names $archives_names
echo "Files loaded to $local_logs_dir"

python3 -m unittest discover

python3 -m experiments.run_experiment \
    --algorithm ci \
    --logs_archives $logs_archives \
    --classes pushups walk sits pullups

# --log_dir $local_logs_dir \
# --classes pushups5_ walk50_ sits10_ typing_

