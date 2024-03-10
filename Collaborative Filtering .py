from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic

# Load dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.2)

# Train KNN model
sim_options = {'name': 'cosine', 'user_based': False}
model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Get recommendations for a user
user_id = '1'
recommendations = model.get_neighbors(int(user_id), k=5)
print("Recommendations for user", user_id, ":", recommendations)
