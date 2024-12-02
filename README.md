# LSVD

Visualize your ideas

## Contributing

- Make a new branch for each feature and create a pr
- Link the issues to your commit for better project overview

## Install the requirements

- [Install git](https://git-scm.com/downloads)
- [Install git lfs](https://git-lfs.com/)
  - Command: git lfs install
- [Install FFMPEG](https://www.ffmpeg.org/download.html)
- [Virtual environment python](docs/Venv.md)

## Jupyter Notebook in google colab

Add these commands to the beginning of your notebook to install all requirements and get the scripts in your google colabs

```shell
# clone our repo and install requirements in colab
%cd /content/
!rm -rf LSVD
!git clone https://github.com/davidg-h/LSVD.git
%cd LSVD
%pip install -r requirements.txt
!apt install ffmpeg
```

You can also just pull our [demo.ipynb](src/demo.ipynb) into colab. It contains the complete code of the project
