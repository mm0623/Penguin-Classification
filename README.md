# Penguin-Classification

<br>

## **1. Group Contributions Statement**

Members: Marissa Meng, Riley Hilt, and Susanna Wu

All three of us wrote the data acquisition and preparation. Susanna led Figure 1, the RandomForestClassifier model and its explaination. Marissa led Figure 2, Table 1, the KNeighborsClassifier model and its explaination. Riley led Figure 3, the SVC model and its explaination. Marissa wrote the function for selecting the best complexity parameter, the function for printing the confusion matrix and classification report with Riley, and the function for plotting decision regions with Susanna. Susanna and Riley wrote the explanation of all figures. We all collaborated for writing the discussion, checked each other’s work, and made revisions to code and writing.

<br>

## **2. Introduction**

Allow us, for a moment, to paint a picture. Imagine a world in which the population of penguins has grown so large that poor researchers in the Antarctic are overwhelmed by the sheer number of penguins; the researchers need assistance determining which features they can use to most easily classify the adorable penguins. In this tutorial, we will create a model that can be conveniently used to categorize penguins by species, specifically, three species: Adelie, Chinstrap, and Gentoo. 

As you can plainly see, the penguins are just precious (Adelie is the cutest), so it should be obvious why we seek to build a program that can help identify the different species. We will use different machine learning algorithms to develop our program by considering and selecting from the following features:

* `Sample number`: the sample number
* `Species`: the species of penguin
* `Region`: the region the penguin was located in
* `Island`: the island the penguin was located on
* `Stage`: the life stage of the penguin
* `Individual ID`: the ID
* `Clutch completion`: the state of the penguin’s nest
* `Date egg`: the date at which the eggs were 
* `Culmen length`: the length of the culmen
* `Culmen depth`: the depth of the culmen
* `Flipper length`: the length of the flipper
* `Body mass`: the mass of the penguin
* `Sex`: the sex of the penguin
* `Delta 15 N`: a measure of the ratio of the two stable isotopes of nitrogen
* `Delta 13 C`: a measure of the ratio of the two stable isotopes of carbon
* `Comments`: any additional comments by the researchers

<br>

## **3. Data Clearning**

shorten penguin species labels, 
Next, import the train_test_split function from sklearn to split the penguins dataframe into training and testing sets, with the size of the training set being 80% of the total penguins dataset and the size of the testing set being 20% of the penguins dataset. The training set will be used in the models themselves, while the testing set will be kept as ‘unseen data’ in order to check the models for overfitting. We will also use this opportunity to set the random state for our data so that the models will produce the same result each time they are run.
Then, the penguin dataset needs to be cleaned and split into the target set y (the penguin species) and the predictor set X (the data which will be used to predict the species). Cleaning includes removing NaN values from the dataset and recoding the qualitative columns of the data to use numerical values. We’ll code this all into a function called clean() and run it on the testing and training sets. 

<br>

# **4. Exploratory Analysis**
Before modeling, we’ll first explore the data to get an idea of which features might prove the most useful in distinguishing between different penguin species. We’ll do this with the pre-cleaned data, so the island, species, and sex columns all use non-numerical values.

### Table 1: Summary table for each island within each species group. 
To examine the data, we can group the penguins by species, then apply the .mean(), .min(), and .max() functions to determine which of these features are distinct to each penguin. From this preliminary analysis, we can determine that of the qualitative features, “Island” is a distinctive feature that isolates different penguin species while “Sex” is not. In addition, culmen length, culmen depth, flipper length, body mass, delta 15 N, and delta 13 C are distinctive quantitative features. Drop the irrelevant columns from the penguins dataset, and create a table with the remaining features (again using the .mean(), .std(), .min(), and .max() functions.)

Further data analysis is necessary in order to choose the best qualitative and two best quantitative features to use. For these, we can create plots in order to eyeball the data and choose the features which yield the most distinct data regions.

### Figure 1: Penguins population for each species on each island.
Firstly, a barplot can be used to evaluate the qualitative data in the penguins dataset. The graph above plots the amount of penguins which are present on each island, separated by species. From the graph we can see that Torgersen island only has Adelie penguins, while Gentoo penguins only live on Biscoe island and Chinstrap penguins only live on Dream island. In addition, there are more than twice as many Gentoo penguins than Adelie penguins on Biscoe island, while there are slightly more Chinstrap penguins than Adelie penguins on Dream island. This means that if an unknown penguin is identified on Biscoe island, it is significantly more likely that the penguin is a Gentoo penguin than an Adelie penguin. If the unknown penguin is identified on Dream island, it is slightly more likely to be a Chinstrap penguin than an Adelie penguin. Finally, if the unknown penguin is identified on Torgersen island, it must be an Adelie penguin.

![figure1](https://github.com/user-attachments/assets/260d9b97-64f1-426d-9277-8625f242ebcb)


### Figure 2: Comparison between each species for every quantitative feature
For the second figure, a box plot allows for clear analysis of the quantitative data. The graph shows the distribution of each feature for each respective species. The graph shows which features have outliers, simplifying the selection of features to use in the machine learning models. Looking at culmen length, the Adelie are considerably smaller than either the Chinstrap or the Gentoo. However, with culmen depth, it is the Gentoo who are shorter than either the Adelie or the Chinstrap. With respect to flipper length and body mass, the Gentoo are larger than the Adelie or the Chinstrap. For delta 15 N, Gentoo are the outlier (smaller), whereas for delta 13 C, it is the Chinstrap who are outliers (larger/less negative). Between the two qualitative features in the penguins dataset, "Island" has a much stronger correlation to penguin species than "Sex" does. In the figure above, it is clear that Torgersen Island only has Adelie penguins, while Gentoo penguins only live on Biscoe Island and Chinstrap penguins only live on Dream Island.

![figure2](https://github.com/user-attachments/assets/f347ce57-170d-4597-b23e-d77e468b335e)


### Figure 3: Pairwise relationship between each quantitative feature
Finally, scatterplots can be used to plot two different quantitative features together, in order to  cross-examine them to determine which two features give the most distinct species regions. Through visual analysis, "Culmen Length" and "Culmen Depth" are good indicators of species, as these graphs show distinct clusters for each species. “Body Mass” and “Flipper Length” plotted against “Culmen Length” also produce nice, distinct clusters. By comparison, “Flipper Length” plotted against “Culmen Depth” produces such overlap that the Adelie and Chinstrap species are practically indistinguishable. In these graphs, two different quantitative features can be cross-examined to determine which give the most distinct species regions. Through visual analysis, "Culmen Length" and "Culmen Depth" shows the most separation between species.

![figure3](https://github.com/user-attachments/assets/d40108de-d8b3-4042-bfe6-0b33684ef330)


#### Gentoo:
* Island - Biscoe
* Culmen depth (<17.3, avg=15.0)

#### Adelie:
* Island - Torgersen
* Culmen length (<46.0, avg=38.8)

#### Chinstrap:
* Island - Dream

<br>

# **5. Feature Selection:**

Based on the average values for culmen length and culmen depth for all three species, it is clear that Gentoo penguins have significantly smaller culmen depth values, while Adelie penguins have significantly smaller culmen length values. In addition, through the earlier analysis of the correlation between penguin species and island, Gentoo penguins only live on Biscoe island while Chinstrap penguins only live on Dream island. Finally, Torgersen island is home to only Adelie penguins. Taking this information and carefully weighing the values of the other features, we ultimately decided to use culmen length, culmen depth, and island to train our models and categorize the penguins.

<br>

# **6. Modeling**

### i. Use cross-validation to choose complexity parameters:
By running 10-fold cross-validation on our models, we can determine the best parameters to use for each model.

To avoid repetitive coding, we can define a function that uses cross-validation to help us pick the complexity parameter that leads to the best average cross-validation score for each of the models. For our purposes, we will be define a function that chooses the best value for a single complexity parameter passed to a KNeighborsClasifier model, a SVC model, or a RandomForestClassifier model.

### ii. Model performance metrics

### iii. Use decision regions to visualize how a model make decisions:
Another helpful way to better understand how our models are making decisions is to plot their decision regions. In other words, decision region plots can help us visualize which data values the model considered to be part of each species. 

In order to plot these regions, we’ll first run the training data through each model, and then split the data by island so each island’s species can be graphed separately. Then, we’ll plot our two quantitative values (culmen length and culmen depth) against each other to get a clear picture of all the data.

### iv. Use confusion matrices to visualize a model's performance:
After we make our models, we want to test their accuracy in predicting the target variable. There are many ways to do this, one of which is by making a confusion matrix. 

A confusion matrix allows use to visualize the performance of a model after it evaluates the data. Here, we only gave the model the testing data to evaluate its performance on unseen data. We visualized the confusion matrix so that darker squares indicates the amount of cross between the actual qualitative value and predicted qualitative value. A perfect prediction would therefore look like dark squares across a diagonal where each square represents a match between the actual and predicted value.

<br>

## Model 1: K Nearest Neighbors Classifier

#### i. Hyperparameter Tuning:
In this model, we’ll run cross-validation on the parameter n_neighbors (the number of neighbors to use for the model.) From the graph, we can see that the optimal hyperparameter k is 1 with the higest 10-fold cross validation score of 98.16%.

#### ii. Fitting and Scoring the Model:
Using the optimal n_neighbors value, we can now create our model, fit it to our training data, and score it against our test data.
Score against training data:  1.0
Score against testing data:  0.9710144927536232
Average score from a 10-fold cross-validation:  0.9428571428571428
From our results, we can see that the model scores consistently high on both the training and test data.

#### iii. Plot the decision regions:
Now, we can plot the model's decision regions to see how its making its decisions.

We see from the above plots that the decisions are not perfect. For instance, in the Dream island plot, one Adelie penguin is incorrectly predicted as a Chinstrap penguin and a Chinstrap penguin is incorrectly predicted as an Adelie penguin. In the Biscoe plot, the model assumes that there are Chinstrap penguins when in fact there are only Adelie and Gentoo penguins. Finally, in the Togersen plot, the model assumes that there are Chinstrap penguins when in fact there are only Adelie penguins. However, these decisions are pretty good as almost all the regions cover the correct corresponding species data points. 

#### iv. Confusion Matrix:
Next, we’ll create a confusion matrix to see where the model made incorrect predictions.

In this matrix, the diagonal of the array represents correct predictions, while the rest of the entries represent incorrect predictions. For example, the entry in the 0th row, 1st column indicates that the model incorrectly predicted the species as Adelie when it was actually a Chinstrap penguin. The two total errors matches the two mismatched points we saw in the decision regions plot. However, it is clear from the plot that these errors are very uncommon, and overall the model was competent at distinguishing between the different penguin species.

#### v. Discussion:
From the decision regions graph, we can see that the model makes several errors in prediction. For example, the graph for Biscoe island includes a large region for Chinstrap penguins when Chinstrap penguins only live on Dream island. In addition, the graph for Dream island includes a region for Gentoo penguins, when Gentoo penguins only live on Biscoe Island. Finally, the graph for Torgersen island includes a small region for Chinstrap penguins as well, when Torgersen island only has Adelie penguins. For the actual data points themselves, however, the model was very accurate in determining which penguins belonged to which species (with only two errors, as seen in the confusion matrix.) With more data to analyze, it’s likely that the nearest neighbors classifier model would be able to predict penguin species with far more accuracy.

<br>

## Model 2: Support Vector Classifier

#### i. Hyperparameter Tuning
For the second model, a support vector classifier (SVC), we’ll run cross-validation on the parameter C (the penalty parameter of the error term, or, how much you want to prevent misclassification). From the graph, we can see that the optimal hyperparameter C is 16 with the higest 10-fold cross validation score of 98.17%.

#### ii. Fitting and Scoring the Model:
Using the optimal C value, we can now create our model, fit it to our training data, and score it against our test data.
Score against training data:  0.9816849816849816
Score against testing data:  0.9855072463768116
Average score from a 10-fold cross-validation:  0.9428571428571428
From our results, we can see that the model scores consistently high on both the training and test data.

#### iii. Plot the decision regions:
Now, we can plot the model's decision regions to see how its making its decisions. The decision regions do a solid job of classifying the data, but not without some minor errors. For example, in the Torgersen graph, one of the Adelie penguins is incorrectly classified as a Chinstrap penguin. In addition, both the Biscoe and Torgersen graphs have an extraneous region for Chinstrap penguins, when no Chinstrap penguins live on either of those islands.

#### iv. Confusion Matrix:
Next, we’ll create a confusion matrix to see where the model made incorrect predictions. Like the previous model's confusion matrix, the diagonal of the array representing correct predictions shows that overall the model was competent at distinguishing between the different penguin species. There is only one incorrect prediction, represented in the third plot that we saw in the decision regions plot. Therefore we can conclude that the model performs well.

#### v. Discussion:
The high accuracy scores and the minimal amount of incorrect predictions as demonstrated by the decision regions and the confusion matrix suggests that the model is quite good in making classification decisions. However, the incorrect predictions indicate that even though the model is fairly accurate, there is still room for improvement. If the model was trained with more data points or different quantitative features, it’s possible that we would get more accurate predictions.

<br>

## Model 3: Random Forest Classifier

#### i. Hyperparameter Tuning
For the third model, we used the random forest classifier model and chose to optimize the parameter max_depth (the maximum level that the decision trees can branch down to) using cross validation.
Best score:  0.9927248677248677
Best max_depth:  11

#### ii. Fitting and Scoring the Model:
Using the optimal max depth value, we can now create our model, fit it to our training data, and score it against our test data.
Score against training data:  1.0
Score against testing data:  0.9855072463768116
Average score from a 10-fold cross-validation:  0.9857142857142858
From our results, we can see that the model scores consistently high on both the training and test data.

#### iii. Plot the decision regions:
Now, we can plot the model's decision regions to see how its making its decisions. From the decision regions graph, we can see that the model was close but not quite perfect. For example, there is a data point on the Dream Island at the boundary between Chinstrap and Adelie which was considered to be a Chinstrap penguin, but is actually an Adelie penguin. Plus, it highlighted a region as Chinstrap despite the entire region on Torgersen Island should be Adelie since only this species is on the island.

#### iv. Confusion Matrix:
Next, we’ll create a confusion matrix to see where the model made incorrect predictions. From the array values, we can see that the model made one error in its prediction when it mistook an Adelie penguin for a Chinstrap penguin. This indicates that the model was extremely accurate when making its predictions, as it was able to correctly predict the species of each penguin every other time.

#### v. Discussion:
The high accuracy scores and the minimal amount of incorrect predictions as demonstrated by the decision regions and the confusion matrix suggests that the model is quite good in making classification decisions. However, the incorrect predictions indicate that even though the model is fairly accurate, there is still room for improvement. If the model was trained with more data points or different quantitative features, it’s possible that we would get more accurate predictions.

<br>

## **7. Concluding Discussion:**
Overall, each model performed extremely well. The nearest-neighbors classifier and the random forest classifier models in particular were able to achieve training scores of 1.0, while the SVC and random forest classifier models only made a single prediction error. However, more data points for each of the penguin species, which would have led to larger training and test sets, would’ve increased the accuracy of each of the models. Having additional columns to select features from would also be helpful. Plus, performing more in-depth feature selection and having the option to use more than just 3 features can lead to better models. In addition, if we can adjust and optimize more than just a single complexity parameter for each model, it’s likely that their performances would also have been improved. 
