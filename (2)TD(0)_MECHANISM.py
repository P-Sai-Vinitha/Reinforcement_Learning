import numpy as np


num_articles = 5
articles = [f"Article {i+1}" for i in range(num_articles)]


transition_matrix = np.random.rand(num_articles, num_articles)
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

reward_matrix = np.random.rand(num_articles, num_articles)


gamma = 0.8         
alpha = 0.1         
num_episodes = 100  
epsilon = 0.1       


V = np.zeros(num_articles)


for episode in range(num_episodes):
    state = np.random.randint(num_articles)
    
    while True:
        
        if np.random.uniform(0, 1) < epsilon:
            next_state = np.random.randint(num_articles)
        else:
            next_state = np.random.choice(num_articles, p=transition_matrix[state])
        
        
        TD_target = reward_matrix[state, next_state] + gamma * V[next_state]
        
        
        V[state] += alpha * (TD_target - V[state])
        
        
        state = next_state
        
        if np.random.rand() < 0.1:  
            break


print("Learned value function:")
for article, value in zip(articles, V):
    print(f"{article}: {value}")
