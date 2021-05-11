## CS440 Spring 2021
## Project 4 - Colorizing
## Mustafa Sadiq (ms3035)

####################################################### IMPORTS #########################################################

from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
from alive_progress import alive_bar

###################################################### FUNCTIONS #########################################################

def save_image(file_name, image_training):
    print('saving image:', file_name)
    Image.fromarray(image_training).save('./REPORT/IMAGES/' + file_name + '.jpg')

def save_image_splitted(file_name, image_training, image_testing=None):
    print('saving image:', file_name)
    Image.fromarray(np.hstack((image_training, image_testing))).save('./REPORT/IMAGES/' + file_name + '.jpg')

def open_image(file):
    image = Image.open(file)
    print('opening image:', file, 'mode:', image.mode)
    return np.array(image)

def split_image(image):
    print('splitting image')
    return np.hsplit(image, 2)

def convert_to_bw(color_image):
    print('converting to bw')
    bw_image = np.copy(color_image)
    x_dim, y_dim = bw_image.shape[0], bw_image.shape[1]
    for x in range(x_dim):
        for y in range(y_dim):
            r, g, b = bw_image[x][y]   
            grayscale =  int(0.21*r) + int(0.72*g) + int(0.07*b)
            bw_image[x][y] = np.array([grayscale, grayscale, grayscale])
    return bw_image

def color_difference_euclidean(a, b):
    return np.linalg.norm(b - a)

def get_patch(image, x, y):
    return np.array([
        image[x][y],
        image[x-1][y],
        image[x+1][y],
        image[x][y-1],
        image[x][y+1],
        image[x-1][y-1],
        image[x-1][y+1],
        image[x+1][y-1],
        image[x+1][y+1]
    ])        

 


##################################################### BASIC AGENT FUNCTIONS ####################################################
def mean_five_colors(image):
    print('finding five mean colors')
    pixels = image.reshape(-1, 3)    
    centroids =  np.array([[250, 0 , 0],
                            [0, 250, 0],
                            [0, 0 , 250],
                            [0, 0, 0],
                            [250, 250, 250]])
    new_centroids = np.copy(centroids)  

    while True:
        centroids = np.copy(new_centroids) 
        point_clusters = {0:np.empty((0, 3), dtype=np.int8), 
                            1:np.empty((0, 3), dtype=np.int8), 
                            2:np.empty((0, 3), dtype=np.int8), 
                            3:np.empty((0, 3), dtype=np.int8), 
                            4:np.empty((0, 3), dtype=np.int8)}

        for pixel in pixels:                  
            minimum_centroid = centroids[0]
            cluster = 0
            minimum_differece = color_difference_euclidean(pixel, minimum_centroid) 
            
            for i in [1, 2, 3, 4]:
                difference = color_difference_euclidean(pixel, centroids[i])
                if difference < minimum_differece:    
                    minimum_differece = difference                    
                    minimum_centroid = centroids[i]            
                    cluster = i                
            point_clusters[cluster] = np.vstack((point_clusters[cluster], pixel))
            # print('\n\n\n\n\n', point_clusters)
        
        # print(point_clusters[0])
        new_centroids[0] = np.mean(point_clusters[0], axis=0)
        new_centroids[1] = np.mean(point_clusters[1], axis=0)
        new_centroids[2] = np.mean(point_clusters[2], axis=0)
        new_centroids[3] = np.mean(point_clusters[3], axis=0)
        new_centroids[4] = np.mean(point_clusters[4], axis=0)

        # print(new_centroids)
        # print('\n\n')

        if np.array_equal(centroids, new_centroids):
            break

    return new_centroids

            
def fill_with_five_colors(image, colors):
    print('filling with five mean colors')
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            minimum = colors[0]            
            minimum_difference = color_difference_euclidean(minimum, image[x][y])
            for i in [1, 2, 3, 4]:
                difference = color_difference_euclidean(colors[i], image[x][y])
                if difference < minimum_difference:
                    minimum = colors[i]
                    minimum_difference = difference
            image[x][y] = minimum  


##################################################### BASIC AGENT RUN ####################################################

def basic_agent():
    file_location = './IMAGES/9.jpg'

    original_image = open_image(file_location)
    save_image('original image3', original_image)

    bw_image = convert_to_bw(original_image) 
    bw_image_training, bw_image_testing = split_image(bw_image)
    save_image_splitted('bw_image_splitted3', bw_image_training, bw_image_testing)

    color_image_training, color_image_testing = split_image(original_image)
    five_colors = mean_five_colors(color_image_training)
    fill_with_five_colors(color_image_training, five_colors)
    save_image_splitted('color_image_splitted_training_five_colored3', color_image_training, color_image_testing)

    print('colorizing image')
    colorized_image = np.copy(bw_image_testing)


    for x in range(bw_image_testing.shape[0]):
        for y in range(bw_image_testing.shape[1]):
            print(x, y)
            if x == 0 or y == 0 or x == bw_image_testing.shape[0]-1 or y == bw_image_testing.shape[1]-1:
                colorized_image[x][y] = np.array([0, 0, 0])
            else:
                current_patch = get_patch(bw_image_testing, x, y)
                patch_difference = dict()

                for z in range(bw_image_training.shape[0]):
                    for w in range(bw_image_training.shape[1]):
                        if z != 0 and w != 0 and z != bw_image_training.shape[0]-1 and w != bw_image_training.shape[1]-1:
                            patch_difference[(z, w)] = color_difference_euclidean(current_patch, get_patch(bw_image_training, z, w))
            
                patch_difference = {k: v for k, v in sorted(patch_difference.items(), key=lambda item: item[1])}           
                patch_difference_coordinates = list(patch_difference.keys())[:6] 
                patch_difference = list(patch_difference.values())[:6]
                            
                data = Counter(patch_difference)
                get_mode = dict(data)
                mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]
                
                if len(mode) != 1:
                    colorized_image[x][y] = color_image_training[patch_difference_coordinates[0][0]][patch_difference_coordinates[0][1]]
                else:
                    colorized_image[x][y] = color_image_training[patch_difference_coordinates[patch_difference.index(mode[0])][0]][patch_difference_coordinates[patch_difference.index(mode[0])][1]]


    save_image_splitted('colorized image3', color_image_training, colorized_image)  




############################### IMPROVED AGENT FUNCTIONS ##########################################

def get_patch_bw(image, x, y):
    return [[
        image[x-1][y+1][0],
        image[x][y+1][0],
        image[x+1][y+1][0],
        image[x-1][y][0],
        image[x][y][0],
        image[x+1][y][0],
        image[x-1][y-1][0],
        image[x][y-1][0],
        image[x+1][y-1][0]
    ]]  



############################### IMPROVED AGENT LOAD DATA ##########################################
def improved_agent_load_data(file_location):   
    original_image = open_image(file_location)

    bw_image = convert_to_bw(original_image) 
    bw_image_training, bw_image_testing = split_image(bw_image)
    

    color_image_training, color_image_testing = split_image(original_image)

    ########################################## bw_image_training_data ######################################################

    bw_image_training_data = np.empty((0, 9), dtype=np.int8)

    for x in range(bw_image_training.shape[0]):
        for y in range(bw_image_training.shape[1]):
            print('bw_image_training_data', x, y)
            if x != 0 and y != 0 and x != bw_image_training.shape[0]-1 and y != bw_image_training.shape[1]-1:
                bw_image_training_data = np.append(bw_image_training_data, get_patch_bw(bw_image_training, x, y), axis=0)
            

    np.save('bw_image_training_data.npy', bw_image_training_data)
    print(bw_image_training_data.shape)

    # ############################################# bw_image_testing_data ###################################################

    bw_image_testing_data  = np.empty((0, 9), dtype=np.int8)

    for x in range(bw_image_testing.shape[0]):
        for y in range(bw_image_testing.shape[1]):
            print('bw_image_testing_data', x, y)
            if x != 0 and y != 0 and x != bw_image_testing.shape[0]-1 and y != bw_image_testing.shape[1]-1:
                bw_image_testing_data = np.append(bw_image_testing_data, get_patch_bw(bw_image_testing, x, y), axis=0)

    np.save('bw_image_testing_data.npy', bw_image_testing_data)
    print(bw_image_testing_data.shape)

    ######################################## color_image_training_data ##############################################

    color_image_training_r_data  = np.empty((0, 1), dtype=np.int8)
    color_image_training_g_data  = np.empty((0, 1), dtype=np.int8)
    color_image_training_b_data  = np.empty((0, 1), dtype=np.int8)

    for x in range(color_image_training.shape[0]):
        for y in range(color_image_training.shape[1]):
            print('color_image_training_data', x, y)
            if x != 0 and y != 0 and x != color_image_training.shape[0]-1 and y != color_image_training.shape[1]-1:
                color_image_training_r_data = np.vstack((color_image_training_r_data, np.array(color_image_training[x][y][0])))                
                color_image_training_g_data = np.vstack((color_image_training_g_data, np.array(color_image_training[x][y][1])))
                color_image_training_b_data = np.vstack((color_image_training_b_data, np.array(color_image_training[x][y][2])))

    np.save('color_image_training_r_data.npy', color_image_training_r_data)
    np.save('color_image_training_g_data.npy', color_image_training_g_data)
    np.save('color_image_training_b_data.npy', color_image_training_b_data)
    print(color_image_training_b_data.shape)

    ############################################ color_image_testing_data ##########################################

    color_image_testing_r_data  = np.empty((0, 1), dtype=np.int8)
    color_image_testing_g_data  = np.empty((0, 1), dtype=np.int8)
    color_image_testing_b_data  = np.empty((0, 1), dtype=np.int8)

    for x in range(color_image_testing.shape[0]):
        for y in range(color_image_testing.shape[1]):
            print('color_image_testing_data', x, y)
            if x != 0 and y != 0 and x != color_image_testing.shape[0]-1 and y != color_image_testing.shape[1]-1:
                color_image_testing_r_data = np.vstack((color_image_testing_r_data, np.array(color_image_testing[x][y][0])))                
                color_image_testing_g_data = np.vstack((color_image_testing_g_data, np.array(color_image_testing[x][y][1])))
                color_image_testing_b_data = np.vstack((color_image_testing_b_data, np.array(color_image_testing[x][y][2])))

    np.save('color_image_testing_r_data.npy', color_image_testing_r_data)
    np.save('color_image_testing_g_data.npy', color_image_testing_g_data)
    np.save('color_image_testing_b_data.npy', color_image_testing_b_data)  
    print(color_image_testing_b_data.shape)   

    print(bw_image_testing.shape)



############################### IMPROVED AGENT ##########################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(predicted, actual):
    return np.power((predicted-actual), 2)



def gradient(predicted, actual, xi):
    return 2 * (predicted - actual) * predicted * (1 - (predicted/255)) * xi

def sgd(learning_rate, weight, input, output, max_iterations):    
    dataX = []
    dataY = []
    # dataY = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

    new_weights = np.copy(weight)    

    def compute():
        for iteration in range(max_iterations):            
            dataX.append(iteration)
            random_sample = np.random.choice(input.shape[0])
            random_input = input[random_sample]
            random_output = output[random_sample]
            predicted = np.dot(random_input, new_weights)
            predicted = sigmoid(predicted)*255
            for i in range(10):      
                gradient_loss = gradient(predicted, random_output, random_input[i])
                new_weights[i] = new_weights[i] - (learning_rate * gradient_loss)

            dataY.append(loss(predicted, random_output))
            yield

    with alive_bar(max_iterations) as bar:
        for i in compute():
            bar()

    plt.plot(dataX, dataY)
    plt.title("Blue model loss with time")
    plt.ylabel('Loss')
    plt.xlabel('Iteration')   
    plt.show() 

    # for i in range(10):
    #     plt.subplot(5, 2, i+1)
    #     plt.plot(dataX, dataY[i])
    #     plt.title("Weight" + str(i))    
    # plt.show()  

    return new_weights

    


def improved_agent():

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
      
    # training data
    bw_image_training_data = np.load('bw_image_training_data.npy')
    color_image_training_r_data = np.load('color_image_training_r_data.npy')
    color_image_training_g_data = np.load('color_image_training_g_data.npy')
    color_image_training_b_data = np.load('color_image_training_b_data.npy')

    bw_image_training_data = bw_image_training_data/255
    bw_image_training_data = np.insert(bw_image_training_data, 0, 1, axis=1)

    # testing data
    bw_image_testing_data = np.load('bw_image_testing_data.npy')
    bw_image_testing_data = bw_image_testing_data/255
    bw_image_testing_data = np.insert(bw_image_testing_data, 0, 1, axis=1)

    learning_rate = 0.00001 # good
    max_iterations = 1000
    weight = np.random.uniform(0.0, 0.001, 10)
    weight_r = sgd(learning_rate, weight, bw_image_training_data, color_image_training_r_data, max_iterations)
    print(weight_r)
    
    # learning_rate = 0.00001 # good
    # # max_iterations = 10000
    weight = np.random.uniform(0.0, 0.001, 10)
    weight_g = sgd(learning_rate, weight, bw_image_training_data, color_image_training_g_data, max_iterations)
    print(weight_g)
    
    # learning_rate = 0.0001
    # # max_iterations = 1000
    weight = np.random.uniform(0.0, 0.001, 10)
    weight_b = sgd(learning_rate, weight, bw_image_training_data, color_image_training_b_data, max_iterations)   
    print(weight_b)

    # return

    predicted_r = np.dot(bw_image_testing_data, weight_r)
    predicted_r = sigmoid(predicted_r)*255

    predicted_g = np.dot(bw_image_testing_data, weight_g)
    predicted_g = sigmoid(predicted_g)*255   

    predicted_b = np.dot(bw_image_testing_data, weight_b)
    predicted_b = sigmoid(predicted_b)*255   



    rgb_uint8 = (np.dstack((predicted_r, predicted_g, predicted_b))).astype(np.uint8)
    output_image = np.reshape(np.ravel(rgb_uint8), (339, 254, 3))


    comparison = (np.dstack((color_image_training_r_data,color_image_training_g_data,color_image_training_b_data))).astype(np.uint8)
    comparison = np.reshape(np.ravel(comparison), (339, 254, 3))    

    save_image_splitted('improved_agent', comparison, output_image)
    






    
############################### IMPROVED AGENT BONUS ##########################################

def improved_agent_bonus():
    from sklearn.neural_network import MLPRegressor  
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    bw_image_training_data = np.load('bw_image_training_data.npy')
    bw_image_training_data = scaler.fit_transform(bw_image_training_data)

    color_image_training_r_data = np.load('color_image_training_r_data.npy')    
    color_image_training_r_data = scaler.fit_transform(color_image_training_r_data)
    color_image_training_r_data = color_image_training_r_data.ravel()

    color_image_training_g_data = np.load('color_image_training_g_data.npy')    
    color_image_training_g_data = scaler.fit_transform(color_image_training_g_data)
    color_image_training_g_data = color_image_training_g_data.ravel()

    color_image_training_b_data = np.load('color_image_training_b_data.npy')    
    color_image_training_b_data = scaler.fit_transform(color_image_training_b_data)
    color_image_training_b_data = color_image_training_b_data.ravel()

    print('Fitting R-model')
    R_model = MLPRegressor(random_state=1, max_iter=500).fit(bw_image_training_data, color_image_training_r_data)
    
    print('Fitting G-model')
    G_model = MLPRegressor(random_state=1, max_iter=500).fit(bw_image_training_data, color_image_training_g_data)
    
    print('Fitting B-model')
    B_model = MLPRegressor(random_state=1, max_iter=500).fit(bw_image_training_data, color_image_training_b_data)
    


    bw_image_testing_data = np.load('bw_image_testing_data.npy')
    bw_image_testing_data = scaler.fit_transform(bw_image_testing_data)


    print('Predicting R-model')
    R_predict = R_model.predict(bw_image_testing_data)

    print('Predicting G-model')
    G_predict = G_model.predict(bw_image_testing_data)

    print('Predicting B-model')
    B_predict = B_model.predict(bw_image_testing_data)

    print(R_predict.shape)    


    rgb_uint8 = (np.dstack((R_predict,G_predict,B_predict))*255).astype(np.uint8)
    output_image = np.reshape(np.ravel(rgb_uint8), (339, 254, 3))

    comparison = (np.dstack((color_image_training_r_data,color_image_training_g_data,color_image_training_b_data))*255).astype(np.uint8)
    comparison = np.reshape(np.ravel(comparison), (339, 254, 3))


    # save_image('bonus_ml', output_image)

    save_image_splitted('bonus_ml', comparison, output_image)



####################################################### ANALYSIS ######################################################################

def analysis():
    original_image_r_data = np.load('color_image_testing_r_data.npy') 
    original_image_g_data = np.load('color_image_testing_g_data.npy') 
    original_image_b_data = np.load('color_image_testing_b_data.npy') 
    original_image_testing = (np.dstack((original_image_r_data,original_image_g_data,original_image_b_data))).astype(np.uint8)
    original_image_testing = np.reshape(np.ravel(original_image_testing), (339, 254, 3))

    colorized_image = open_image('./REPORT/FINAL IMAGES/bonus_ml1.jpg')

    # _, original_image_testing = split_image(original_image)
    _, colorized_image_testing = split_image(colorized_image)

    # print('black border on original_image_testing')
    # for x in range(original_image_testing.shape[0]):
    #     for y in range(original_image_testing.shape[1]):
    #         if x == 0 or y == 0 or x == original_image_testing.shape[0]-1 or y == original_image_testing.shape[1]-1:
    #             original_image_testing[x][y] = np.array([0, 0, 0])

    save_image('comparisonOriginal', original_image_testing)
    save_image('comparisoncolorized', colorized_image_testing)


    print('MSE:', np.square(np.subtract(colorized_image_testing, original_image_testing)).mean())
    






############################################################
# improved_agent_load_data('./IMAGES/9.jpg')
# improved_agent()
# improved_agent_bonus()
# analysis()
