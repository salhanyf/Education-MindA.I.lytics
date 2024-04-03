# Education-MindA.I.lytics
COMP 472 Artificial Intelligence <br>
By Group AK_17 <br>

Farah Salhany <br>
Aleksandr Vinokhodov <br>
Athiru Pathiraja <br>

Project repository
https://github.com/salhanyf/Education-MindA.I.lytics

Link to Dataset 
https://github.com/salhanyf/Education-MindA.I.lytics/tree/main/Dataset

Dataset used in our project was taken from a public library Kaggle with CC0 Public Domain licensing terms:
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

This archive should include the following items: <br>
01 - Dataset samples folder: includes 25 representative images from each class in our dataset. <br>
02 - README.txt file: general description of the submission including team members information and IDs. <br>
03 - Team_AK_17_Expectations-of-Originality file: a signed expectation of originality form for each team members. <br>
04 - Team_AK_17_Part2-report file: Part 2 report of this project. <br>


Steps to run code: <br>
1. to create training, validation, and testing datasets from your raw data, use the split_dataset function, specifying the path to your raw dataset, the train ration, val ratio, test ratio and the random_state input parameters. 
2. to train a model, use the right intialization and forward method in FacialImageCNN module. For eg, if you would like to train a model with 2 convolution layers, and a kernel size of 3x3, then use the appropriate init and forward methods, commenting the rest. Then create an instance of the class, and use the train_model function. Ensure that you have specified DataLoaders for the train_loader parameters and # of epochs for 'epochs' parameter. You may save the model after training using the 'save_model' command, specifiying the directory for it to be saved.
3. To load a model, use the 'load model' parameter, specifying the the path to the model you would like to load.
4. To evaluate a model, use the 'evaluate_model' function, specifying the loaded model, the data loaders, and classes as input parameters.
