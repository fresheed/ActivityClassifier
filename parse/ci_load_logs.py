import dropbox
import shutil
import os
from argparse import ArgumentParser
import tarfile


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--archive_name", required=True)
    parser.add_argument("--local_dir", required=True)
    args=parser.parse_args()

    client=dropbox.Dropbox(args.token)

    local_dir=args.local_dir
    try:
        shutil.rmtree(local_dir)
    except FileNotFoundError:
        pass
    os.mkdir(local_dir)

    archive_name=args.archive_name
    local_path=os.path.join(local_dir, archive_name)
    with open(local_path, "wb") as local_file:
        remote_path="/"+archive_name
        metadata, data =client.files_download(path=remote_path)
        local_file.write(data.content)

    with tarfile.open(name=local_path, mode="r|gz") as archive:
        archive.extractall(path=local_dir)
