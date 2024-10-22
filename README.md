
**Introduction:**
 	In the realm of food and nutrition, meal planning often presents a daunting challenge, requiring individuals to balance various factors such as taste, nutrition, dietary restrictions, and personal preferences. This complexity can lead to stress, time-consuming meal preparation, and difficulties in adhering to dietary goals. Moreover, there is a lack of inclusive social media platforms that cater to individuals of all ages, genders, and abilities, depriving them of a supportive community to share culinary experiences and insights.
To address these pressing issues, our team embarked on the development of ChefGPT—an innovative AI-based interactive chatbot and a dedicated food community social media platform equipped with an enhanced recommendation system. With a mission to provide personalized assistance in cooking and foster an inclusive food community, ChefGPT leverages cutting-edge technologies to offer tailored recommendations and facilitate meaningful connections among users.

Problem Domain: Food & Nutrition


**Problem Statement:**

To develop an AI-based interactive chatbot for personalized assistance in cooking along with a specially dedicated food community social media site with an enhanced collaborative filtering-based recommendation models compatible with all users irrespective of age, gender and disabilities
Problem 1:
Many individuals face challenges in meal planning due to the complexities of balancing nutrition, taste, dietary restrictions, and personal preferences. This often leads to stress, time-consuming meal preparation, and difficulties in adhering to dietary goals.
Problem 2:
In addition to the challenges in meal planning, we also head towards tackling the problem of not having a social media platform for all people irrespective of their age, gender and disabilities

Development Environment:
HTML, CSS,NodeJS, LLM, FastAPI/Python, Firebase, Ngrok


**Steps in building recommendation system: **
	
The following are the steps involved in building a food recommendation system.

Data collection and basic preprocessing
ML Model Training and Testing:
Content based recommendation(Cosine similarity):
Collaborative filtering:
Popularity based recommendation
ANN (Neural Collaborative Filtering) 
KMeans
Combining recommendations
Export and Deployment 

1.Data Collection and Preprocessing:
	In order to build a recommendation system we have to have an initial dataset of existing users, food items and user interactions for a particular. But unfortunately there is no clear dataset available as such. So we have used 5000 Indian Cuisines Dataset(with images) (kaggle.com) to get detailed information about food name, ingredients, recipe, prep time, etc. 

Data set description:
The data was created to build a deep learning based image to recipe model which can provide ingredients and recipe of a dish once the image is uploaded.The csv data was scraped from https://www.archanaskitchen.com/ and the images were downloaded using a image downloader python code. 


Having this dataset as the base, we created a synthetic dataset for users and user interactions with food items using Faker modules.
Preprocessing techniques like checking for null and missing values and removing duplicates and invalid URL IDs were also performed.


2. ML Model training and testing:

Cosine similarity (Content-based recommendation):
This type of recommendation uses TF-IDF vectorization to convert combined textual features into numerical vectors. Then, it calculates cosine similarity between these vectors and finds close matches for a given food name using difflib.




Collaborative filtering:
Popularity based recommendation:
This script analyzes user interactions with food items, calculating popularity based on swipes and likes. It then selects the top N popular food items as recommendations, providing a simple but effective way to suggest popular choices to users.

K Means:
The process begins with data preparation, encoding user and food IDs into continuous indices for efficient handling. A Matrix Factorization model is then constructed using PyTorch, initialized with embeddings for users and food items. To facilitate training, PyTorch's DataLoader is employed, enabling batch processing over multiple epochs. Mean Squared Error loss and the Adam optimizer are chosen for model training, iteratively refining latent factors to improve predictive accuracy. Post-training analysis involves extracting and interpreting latent factors for both users and food items, providing insights into user preferences and food characteristics. Additionally, k-means clustering is applied to food embeddings, grouping similar food items to enhance recommendation systems and comprehend food preferences on a deeper level. This comprehensive approach integrates data preprocessing, model training, and post-training analysis to build a robust recommendation system tailored to food preferences.

ANN (Neural Collaborative Filtering - NCF) :
This type of ML modeling involves building and training a recommendation model using neural networks for a food recommendation system. Categorical variables like user IDs and food IDs are encoded using label encoding. The dataset is split into training and testing sets for model evaluation. The neural network architecture includes embedding layers for both users and foods, followed by flattening and concatenation. Dense layers with ReLU activation functions are used for feature transformation, and the final output layer predicts whether a user will like a given food item using a sigmoid activation function. The model is compiled with binary cross-entropy loss and Adam optimizer. After training the model, it is saved and then loaded for making recommendations. A function is defined to generate predictions for all food items for a specified user, utilizing the loaded model and label encoders. Finally, recommendations are sorted based on the predicted probabilities, providing personalized food recommendations for the user.




Combining recommendations:
FastAPI endpoint for K Means model is created to integrate two recommendation strategies. Firstly, it utilizes k-means clustering to suggest two unseen food items from the same cluster as the input food ID, filtered based on user interactions. Secondly, it recommends the latest posts related to these food items, combining content-based filtering with user interaction data to offer personalized suggestions.
Another FastAPI endpoint for NCF is created and recommendations are generated using two approaches: collaborative filtering with neural networks and popularity-based sorting. After filtering out foods already interacted with by the user, a subset of recommended foods is randomly selected from both recommendation lists. This combination ensures a blend of personalized suggestions and popular choices, enhancing the diversity and relevance of the final recommendations.
Additionally, requests are sent to both the endpoints and randomly 5 recommendations are considered.



Export and Deployment:
The provided FastAPI code establishes a RESTful API for deploying a machine learning model. Utilizing ngrok, the local server is exposed securely, enabling external access. The ann_prediction endpoint orchestrates model inference, merging collaborative filtering and popularity-based recommendations for personalized results. Additionally, K-Means prediction endpoint retrieves the latest post related to these food items, providing a blended recommendation approach based on user interactions and content relevance.



**Comparison and Inference: **

Measure Score 
K Means
Neural Collaborative Filtering
Average
Training RMSE
0.021155332658730763
0.3167308675386013
0.1689431
Testing RMSE
0.021132196508323445
0.20811172965789476
0.114621963
Training MAE
0.01659827105106203
0.34735194557116617
0.181975109
Testing MAE
0.01658194564374025
0.22923666960293734
0.122909307


Table 1: Comparison of evaluation metrics (RMSE and MAE)					

The proposed system talks about combining the results of K Means clustering and Neural Collaborative filtering along with popularity-based recommendation. So we can compare each model’s RMSE and MAE with the combined average. We can infer from Table 1 that averaging the results yields very low RMSE and MAE values and using the enhanced collaborative model gives better results for making recommendations.


**Sustainable Goals Targeted**

1. SDG 2: Zero Hunger
2. SDG 3: Good Health and Well-being
3. SDG 12: Responsible Consumption and Production
4. SDG 10: Reduced Inequalities
5. SDG 17: Partnerships for the Goals

**Target Audience**

Individuals facing time constraints in meal planning and preparation.
Consumers with specific dietary requirements or allergies.
Health-conscious individuals or fitness enthusiasts pursuing balanced nutrition.
Those seeking innovative culinary solutions for meal variety and leftovers.
People with disabilities get an experience on online social media.

Conclusion and Future Works:
The proposed system talks about combining the results of K Means clustering and Neural Collaborative filtering along with popularity-based recommendation. So we can compare each model’s RMSE and MAE with the com

Learning Outcomes:
Understood and implemented K Means Clustering, Neural Networks
Understood and implemented Cosine similarity, matrix factorisation.
Understood the concept of preprocessing techniques and implemented it.
Understood and implemented cloud deployment and exporting of ML models. 

References:
Thongsri N, Warintarawej P, Chotkaew S, Saetang W. Implementation of a personalized food recommendation system based on collaborative filtering and knapsack method. Int. J. Electr. Comput. Eng. 2022 Feb 1;12(1):630-8.
Naik PA. Intelligent food recommendation system using machine learning. Nutrition. 2020 Aug;5(8).
Gkatzola K, Papadopoulos K. Social media actually used by people with visual impairment: A scoping review. British Journal of Visual Impairment. 2023:02646196231189393.
Wu S, Adamic LA. Visually impaired users on an online social network. InProceedings of the sigchi conference on human factors in computing systems 2014 Apr 26 (pp. 3133-3142).
Khasawneh MA. Digital inclusion: Analyzing social media accessibility features for students with visual impairments. Studies in Media and Communication. 2023;12(1):71.
Cui Y, Liang S, Zhang Y. Multimodal representation learning for tourism recommendation with two-tower architecture. Plos one. 2024 Feb 23;19(2):e0299370.
Gong S. A collaborative filtering recommendation algorithm based on user clustering and item clustering. J. Softw.. 2010 Jul 1;5(7):745-52.









