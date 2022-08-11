# Feaderated Learning Demonstrator

## Project description
Welcome to the Federated Learning Demonstrator which has some necessary function to explain the functionality of federated learning on your own browser. Following functions can be executed: a client can be started, a server can be started, local data can be generated and saved, client can participated in a federated learning round, demonstrator shows result of every federated round, classification of new data with federated (global) model can be executed.


## How to contribute
### Project structure
```shell
├── .git/
├── fit_global_model/
├── pages/
│ ├── 1_Explanation.py
│ ├── 2_What_to_do?.py
│ ├── 3_1._Data_Generation.py
│ ├── 4_2._Participate_in_Federated_Training.py
│ ├── 5_3._Test_the_federated_Model.py
│ ├── 6_4._Results.py
├── pictrues/
├── .gitignore
├── client.py
├── client2_own_data_generated.py
├── client3_with_MNIST_data.py
├── image_data.npz
├── README.md
├── requirements.txt
├── server_without_streamlit.py
```
### Git branch and merge
For contributing to the project please check out your own branch and send a merge request.
## How to use the Demonstrator
### Prerequisites
Python with the version 3.9 is necessary. Flower with the version  0.19.0 is used for federated learning framework. Streamlit with the version 1.10.0 is used for the GUI. You can install this packages with the following command.

```shell
pip install flwr streamlit
```
### Setup
To set up and run this project you have to clone the project with this command:
```shell
git clone https://github.com/ludewi/dev_demonstrator.git
```
Install the required packages with:
```shell
pip install -r requirements.txt
```

### Run
Activate your python environment with:
```shell
source /path/to/venv/bin/activate
```
### How to use Demonstrator
Start the demonstrator with 
```shell
streamlit run client.py
```

Start the Server with 
```shell
python server_without_streamlit.py
```

Start other clients without GUI with
```shell
python client2_own_data_generated.py
```
```shell
python client3_with_MNIST_data.py
```