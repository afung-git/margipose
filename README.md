# MargiPose

Accompanying PyTorch code for the paper
["3D Human Pose Estimation with 2D Marginal Heatmaps"](https://arxiv.org/abs/1806.01484).

## Setup

Requirements:

Install Dependecies
* pip install -r requirements.txt

### Prepare datasets

You only need to prepare the datasets that you are interested in using.

NOTE: Currently `docker-compose.yml` contains example volume mounts for the datasets.
You will need to edit the entries for datasets that you have prepared, and remove
the others.

#### Human3.6M

1. Use the scripts available at https://github.com/anibali/h36m-fetch to download
   and preprocess Human3.6M data.
2. Edit the volume mounts in `docker-compose.yml` so that the absolute location of
   the `processed/` directory created by h36m-fetch is bound to `/datasets/h36m`
   inside the Docker container.

#### MPI-INF-3DHP

1. Download [the original MPI-INF-3DHP dataset](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).
2. Extract content of the zipfile
3. cd to root directory
4. Modify the conf.ig file by setting download location and setting the ready_to_download flag = 1
5. set permission to execute get_dataset.sh with sudo chmod +x ./get_dataset.sh
6. set permission to execute get_get_testset.sh with sudo chomod +x ./get_testet.sh
7. Run ./get_dataset.sh
8. Run ./get_testset.sh

9. Use the `src/margipose/bin/preprocess_mpi3d.py` script to preprocess the data.
11. Edit the Base_Data_Dir environment variable in the get_dataset.py file to reflect the location of the extracted dataset

#### MPII

1. Edit the volume mounts in `docker-compose.yml` so that the desired installation directory
   for the MPII Human Pose dataset is bound to `/datasets/mpii` inside the Docker container.
2. Run the following to download and install the MPII Human Pose dataset:
   ```
   $ ./run.sh python
   >>> from torchdata import mpii
   >>> mpii.install_mpii_dataset('/datasets/mpii')
   ```

### [Optional] Configure and run Showoff

Showoff is a display server which allows you to visualise model training progression.
The following steps guide you through starting a Showoff server and configuring
MargiPose to use it.

1. Change `POSTGRES_PASSWORD` in `showoff/postgres.env`. Using a randomly generated password is
   recommended.
2. Change `COOKIE_SECRET` in `showoff/showoff.env`. Once again, using a randomly generated
   value is recommended.
3. From a terminal in the showoff directory, run `docker-compose up -d showoff`. This will
   start the Showoff server.
4. Open [localhost:13000](http://localhost:13000) in your web browser.
5. Log in using the username "admin" and the password "password".
6. Change the admin password.
7. Open up `showoff/showoff-client.env` in a text editor.
8. From the Showoff account page, add a new API key. Copy the API key ID and secret key
   into `showoff-client.env` (you will need to uncomment the appropriate lines).

## Running scripts

Train a MargiPose model on the MPI-INF-3DHP dataset:

```bash
./run.sh margipose train with margipose_model mpi3d
```

Train without pixel-wise loss term:

```bash
./run.sh margipose train with margipose_model mpi3d "model_desc={'settings': {'pixelwise_loss': None}}"
```

Evaluate a model's test set performance using the second GPU:

```bash
./margipose/bin/eval_3d.py --model pretrained/margipose-mpi3d.pth --dataset mpi3d-test
```

Explore qualitative results with a GUI:

```bash
./margipose/bin/run_gui.py --model pretrained/margipose-mpi3d.pth --dataset mpi3d-test
```

Image Inference:

```bash
./infer.py --model pretrained/margipose-mpi3d.pth --mode i --path inputs/000001.jpg
```
or for a dir of inputs

```bash
./infer.py --model pretrained/margipose-mpi3d.pth --mode d --path inputs/
```

Video Inference:
```bash
./infer.py --model pretrained/margipose-mpi3d.pth --mode v --path inputs/videos/000001.mp4
```

Run the project unit tests:

```bash
./setup.py pytest
```

## Pretrained models

Pretrained models are available for download:

* [margipose-mpi3d.pth](https://cloudstor.aarnet.edu.au/plus/s/fg5CCss8o9PdURs) [221.6 MB]
  * Trained on MPI-INF-3DHP and MPII examples
* [margipose-h36m.pth](https://cloudstor.aarnet.edu.au/plus/s/RisOjU8YwqUXFI7) [221.6 MB]
  * Trained on Human3.6M and MPII examples

Once downloaded place them in the /pretrained directory
## License and citation

(C) 2018 Aiden Nibali

This project is open source under the terms of the Apache License 2.0.

If you use any part of this work in a research project, please cite the following paper:

```
@article{nibali2018margipose,
  title={3D Human Pose Estimation with 2D Marginal Heatmaps},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1806.01484},
  year={2018}
}
```
