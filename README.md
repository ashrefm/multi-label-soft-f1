# Overview
This is a tutorial illustrating how to build and train a machine learning system for multi-label image classification with TensorFlow 2.0.

For example, can we predict the genre of a movie just from its poster ? We will be using a movie poster dataset hosted on Kaggle. Given an image of a movie poster, the model should predict one or many correct labels (Action, Romance, Drama, etc.).

We can download a pre-trained feature extractor from TensorFlow Hub and attach a multi-headed dense neural network to generate a probability score for each class independently.

The model is trained in two ways: the classic "binary cross-entropy" loss is compared to a custom "macro soft-F1" loss designed to optimize directly the "macro F1-score". The benefits of the second method are demonstrated to be quite interesting.

Please, check these two blog posts for a full description:
* Multi-Label Image Classification in TensorFlow 2.0
* The Unknown Benefit of using a Macro Soft-F1 Loss in Classification Systems


# Install
The required Python packages for executing the scripts in this repository are listed in `requirements.txt` and `requirements_gpu.txt`.  
We recommand using Python >= 3.5 and the standard virtualenv tool.  

You may need to ugrade the Python package manager pip before installing the required packages:
```
$ pip install --upgrade pip
```

At the terminal, run the following command to create a virtual environment.
```
$ virtualenv tf2env
```

Activate the environment: 
```
$ source tf2env/bin/activate
```

Install the necessary python packages (use the second command line for tensorflow-gpu)
```
$ pip install -r requirements.txt
$ pip install -r requirements_gpu.txt
```

Check the list of packages installed and that you have TensorFlow 2.0 among them.
```
$ pip list
```

Add Tensorflow 2 virtual environment to Jupyter:
```
$ pip install ipykernel
$ python -m ipykernel install --user --name=tf2env --display-name "TensorFlow 2"
```

Launch Jupyter Notebook:
```
$ jupyter notebook
```

If you need to delete this environment later:
```
$ jupyter kernelspec uninstall tf2env
$ rm -rf tf2env
```


# Data
The dataset is hosted on [Kaggle](https://www.kaggle.com/neha1703/movie-genre-from-its-poster) and contains  movie posters from [IMDB Website](https://www.imdb.com/). From there, we can get a csv file `MovieGenre.csv` with the following information for each movie: IMDB Id, IMDB Link, Title, IMDB Score, Genre and link to download the movie poster.  
In this dataset, each Movie poster can belong to at least one genre and can have at most 3 genre labels assigned to it.
We recommend using a function called `download_parallel` that was prepared in the `utils.py` module. This helps speed up the download of the image dataset. Check the tutorial notebook on how to use the function in association with the original csv file.

<img src="./img/posters_2x4.png" width="900">

# Workflow
You can walkthough the [tutorial notebook](https://github.com/ashrefm/multi-label-soft-f1/blob/master/Multi-Label%20Image%20Classification%20in%20TensorFlow%202.0.ipynb) and execute the following steps:
* Data collection
* Data preparation
* Create a fast input pipeline in TensorFlow
* Build up the model
    * Get a transfer learning layer using TensorFlow Hub  
    * Stack a multi-label neural network classifier on top
* Model training and evaluation
* Understand the role of macro soft-F1 loss
* Export and save tf.keras models


# Resources
* [TensorFlow Tutorial on Transfer Learning with TF.HUB](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)
* [François Chollet's Tutorial on Transfer Learning with a pre-trained ConvNet](https://www.tensorflow.org/tutorials/images/transfer_learning)