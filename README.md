# ironhack_project_story_of_football
This is my mid bootcamp project, where I carried out exploratory data analysis of Europe's top 5 leagues using Tableau. And also creating a logistic regression model to predict outcome of a match. 

1. On this branch, there is one folder called project and that contains all the files
2. Inside the project folder, there are multiple subfolders:
- _data: contains all the datasets used or created in this project. The raw data are understat_per_game.csv and understat_com_updt.csv. 
Theres also datasets I updated/created myself without the russian league. understat_per_game_updated.csv and understat_com_updt.csv
Theres also a dataset which has the top3 teams called best_team.csv
And you can find X_train, y_train, X_test, y_test, X_train_trans,X_test_trans, X_train_SMOTE, y_train_SMOTE which have been used for the model
And finally a zip file containing raw data

-_functions: .py file for various purposes:
num_cat_splitter: which takes the uderstat_per_game.csv (as a dataframe on pandas) and seperates into numerical and categorical.
min_max: takes only numerical dataframes and normalizes it using min max scaler from sklearn
label_encoder: takes categorical ordinal data and label encodes it using label encoder from sklearn
oh_encoder: takes categorical nominal data and encodes it using OneHotEncoder from sklearn
cleanin_football: needs both understat_com and understat_per_game, and cleans it by renaming first two columns and getting rid of Russian League data, it deals with duplicates as well

-_images: images used in tableau presentation, including confusion matrixes

-_jupyter_notebooks: contains 5 files. 
football_story_data_exploration_cleaning.ipynb can be used to see how I did data exploration and cleaning. 
football_story_modelling.ipynb can be used to see how I did modelling with various upsamling/downsampling method, as well as how I saved my functions and model. 
main.ipynb shows how I imported the functions I made and how I use them on a test file
main.py is the .py file of the main.ipynb

-_model: contains my model saved with joblib. Please use joblib to read  the model. It has been fitted with X_train_SMOTE and y_train_SMOTE(these can be found in data foler)

-_tableau: contains tableau workbook for the project. EDA and modelling resuls displayed here. 

-_transformers: labelencoder_catord.sav, minmaxscaler_numericals.sav, onehotencoder_catnom.sav were all saved using Pickle. So use pickel to load them. They have been fitted with X_train data. 


link to tableau workbook just incase:

https://public.tableau.com/views/mid_project_aaron_pereira/Story1?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

