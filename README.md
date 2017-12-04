# TSNE-Embedding-Visualisation
A Simple and easy to use way to Visualise Embeddings!
Blog post on this project is [here](https://buzzrobot.com/using-t-sne-to-visualise-how-your-deep-model-thinks-4ba6da0c63a0).
<p align="center">
  <img src="https://github.com/harveyslash/TSNE-Embedding-Visualisation/blob/master/demo.gif?raw=true" alt="Visualising Example"/>
</p>




### What this project is? 
This project is forked from [Tensorflow's Standalone Embedding Projector](https://github.com/tensorflow/embedding-projector-standalone).
It shows how a pretrained InceptionV3 model can be used on images and plotted in an interactive 3d map.


### Why this over the Standalone Projector? 
This project allows you to visualise any array of vectors with a light depency stack. It is designed to be decoupled from any library. Moreover , it uses a static file system, so you can publish your results without requiring a server. E.g. https://harveyslash.github.io/TSNE-Embedding-Visualisation/.

### Project Structure

    |-- data  <-- where to put raw data
    |-- Feature-extractor.ipynb <-- Demo of Embedding generation in a step by step fashion
    |-- index.html <-- The GUI of the Viewer (Do not touch, unless you know what youre doing)
    |-- LICENSE
    |-- main.py <-- Executable to generate embedding data from command line args
    |-- oss_data <-- required by the visualisation project
    |   |-- oss_demo_projector_config.json <-- all configuration files are stored here, this is modified by main.py automatically
    |   |-- sprites.png <-- sprites for the demo 
    |   `-- tensor.bytes <-- embeddings array for the demo
    `-- requirements.txt

### Installation and requirements
This project requires python3.6. You can install all dependencies using `pip install -r requirements.txt`

### Usage 
Usage: main.py [OPTIONS]

    Options:
      --data TEXT                 Data folder,has to end with /
      --name TEXT                 Name of visualisation
      --sprite_size INTEGER       Size of sprite
      --tensor_name TEXT          Name of Tensor file
      --sprite_name TEXT          Name of sprites file
      --model_input_size INTEGER  Size of inputs to model
      --help                      Show this message and exit.
  
### Visualising
After you have run main.py, the sprites, tensors, and config.json should be updated. You can then serve the visualisation using a static file server. I just run `python -m SimpleHTTPServer` at the root level of the project. You can also upload the files to a github repository and then view it using github pages. 
