# Development (Linux)

These are the steps to setup a development environment for Linux after cloning this repository:

1. Create a python virtual environment

```
cd <root_directory_of_this_repository>
python3 -m venv .venv
```

Make sure that you activate this python environment before running a python script of this repository:

```
cd <root_directory_of_this_repository>
. .venv/bin/activate
```

2. Install the python requirements

```
cd <root_directory_of_this_repository>
pip install -r requirements.txt
```

3. Download the CelebA dataset and decompress it under the data/CelebA-HQ folder

You'll find the dataset in the following links:

- [Github repository](https://github.com/switchablenorms/CelebAMask-HQ)
- [Google Drive - downloading link](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

```
cd <root_directory_of_this_repository>
mkdir data/CelebA-HQ/images
mv ~/Downloads/CelebAMask-HQ.zip data/CelebA-HQ/images/.
unzip data/CelebA-HQ/CelebAMask-HQ.zip
```
