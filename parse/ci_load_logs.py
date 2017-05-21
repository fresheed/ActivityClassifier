import dropbox
import shutil
import os
from argparse import ArgumentParser


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--local_dir", required=True)
    args=parser.parse_args()

    client=dropbox.Dropbox(args.token)

    local_dir=args.local_dir
    try:
        shutil.rmtree(local_dir)
    except FileNotFoundError:
        pass
    os.mkdir(local_dir)

    for entry in client.files_list_folder('').entries:
        name=entry.name
        if not name.endswith("_log"):
            continue
        local_path=os.path.join(local_dir, name)
        with open(local_path, "wb") as local_file:
            remote_path="/"+name
            metadata, data =client.files_download(path=remote_path)
            local_file.write(data.content)
