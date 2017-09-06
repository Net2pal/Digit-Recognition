# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/temp/data/",one_hot=True)

import tensorflow as tf

#Set parameter
learning_rate = 0.01
#learning rate define how fats we want to update our weights if leaning rate is to big, our model might skip the optimal solution & if it is to small we might nee too many interaction to coverage on best result.    0.001 because it known as decent learning rate foe this problem
training_iteration = 30
batch_size = 100
display_step=2

#Tf graph input
x = tf.placeholder("float",[None,784])#mnist data image of shape 28*28
y = tf.placeholder("float",[None,10])#0-9 digit recognition

#create a model

#set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#biases for model as biases let a shift our regression line to better fit the data
	
with tf.name_scope("Wx_b") as scope:
  
  #scope help us organize modes in the modes in the graph visualizer called tensor board which will view at the ends
	#constaruct a linear model
  model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
	
#Add Summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

#more name scopes will claen up grap represenation
with tf.name_scope("cost_function") as scope:
	#cost fuction help in minimize erroe using cross entropy
	  #cross entropy
	
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)
with tf.name_scope("train") as scope:
   	#gradient descent which take learning_rate as a parameter
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
	
#initializing the variable
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

## Launch the graph
with tf.Session() as sess:
    sess.run(init)

    
    
    # Change this to a location on your computer
    summary_writer = tf.summary.FileWriter('C:\\Users\\Pulkit\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\tensorflow\\python\\summary\\writer', sess.graph)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print ( "Tuning completed!" )

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ( "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
