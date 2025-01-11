# ml-assignment-project
A simple ML project with a CI/CD pipeline.

# Hyperparameter tuning and choosing best performing parameters
![img_4.png](img_4.png)

![img_5.png](img_5.png)

# Code for hyperparameter tuning using **GridSearchCV**
![img_6.png](img_6.png)

### Hyperparameters passed
![img_7.png](img_7.png)

# Docker image building and running steps:

docker build -t ml-flask-app .
docker run -p 5050:5050 ml-flask-app

### Docker image
![img.png](img.png)

### Docker container running flask api

![img_1.png](img_1.png)


### Docker run command running flask api for model serving
![img_2.png](img_2.png)

### API response for requested features

![img_3.png](img_3.png)