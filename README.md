🧠 MNIST Neural Network from Scratch (NumPy Only)
This project is a minimal yet complete implementation of a feedforward neural network trained on the MNIST dataset (handwritten digits 0–9), using only NumPy — no external machine learning libraries like TensorFlow or PyTorch were used.

The goal is to understand and demonstrate the fundamental mechanics of neural networks, including forward propagation, backpropagation, and gradient descent.

🚀 Features

    Implemented completely from scratch using Python + NumPy

    Trains a simple fully-connected neural network  on the MNIST dataset

    Supports:

        Configurable number of hidden layers and neurons

        Sigmoid or ReLU activation

        Softmax output with cross-entropy loss

        Mini-batch gradient descent

    Accuracy tracking during training


📊 Dataset

The MNIST dataset contains 60,000 training and 10,000 testing images of handwritten digits (28x28 pixels). The dataset is loaded and preprocessed into flat 784-dimensional vectors.


🧮The whole idea of the neuron network

Now an image is a 28x28 pixels, so in tolat we have 784 pixels.
The pixles takes value between 0-255(where o => balck, 255 => white )
we have a n amount of pictures so we can represent this :

        ┌───────────────784 features───────────────┐
    X = │  x₁₁   x₁₂   x₁₃   ...   x₁₇₈₄ │  ← Image 1
        │  x₂₁   x₂₂   x₂₃   ...   x₂₇₈₄ │  ← Image 2
        │   .     .     .    ...     .   │
        │   .     .     .    ...     .   │
        │  xₙ₁   xₙ₂   xₙ₃   ...   xₙ₇₈₄ │  ← Image n
        └─────────────────────────────────────────┘
            ↑
       Each row is a flattened 28×28 image

Now the above matric we need to transpouse, this is done because when we multiply with the weight matrix and add the bias (Z=W⋅X+b), it is better(some algebra big brain i do not know, i fail the algebra in school)


And we need to build the neuron network, and it will consist of 3 layers (0,1,2),
here the first layer(layer 0) is the input layer here we will have 784 neurons(because this is the amount of values each image is represented), then we have the first deep layer which will have 10 neurons(just because we can)
and last we will have 10 neurons at the output layer, here we need this much neurons because in this layer we will determin what is the correct number(and we have 10 possible numbers)

The network will be fully conected so every neuron will go to every neuron in the next layer, and we will have to implement a kind of cycle which will consist of:

1.forward pass

2.backpropagation

3.update parameters

And this is going to go in a cycle: forward pass => backpropagation => update parameters => forward pass and so on ,  up to the point where the network has "learn"!

Now for the implemetation of all of this(to implement a neuron network is kind of abstrack, we will just do the math part, adn that thing with small balls is just to understand it ):

Let's start with the forward pass:
Here is simple in fact, we will multiply the weight with the values(the array of numbers we have from the img) and add bias Z = W x X + b
And do this for all the neurons in every layer(more presise it will be from 0 => 1 and from 1 => 2).
But here we need also sth more a aactivaton function. If we do not add this the function that we will get will be linear adn this is not so nice!
In here we will apply two activation function :
 in the second layer(the 1 deep layer) => apply a ReLU function

 ReLU = {x if x > 0 and 0 otherwise

and at the last layer we applay a softmac funciton this is done because this function will give as values for all the neurons(10 neurons) that will add all to 1
Example: n1=1, and all the others will be 0(n2, n3, n4 ..... = 0), this is just theorical because this is too perfect.In reality we will get values like 0.8, 0.02 etc


🔁 Backpropagation — How the Network Learns

Backpropagation is the algorithm that lets a neural network learn from its mistakes. After the network makes predictions in the forward pass, backpropagation is how it adjusts its weights to do better next time — using a method called gradient descent.

Now this is the beauty part(not really this envolves derivates and remember i fail algebra, but also calculus -_- )

Here we start from the end, first we see how much we miss the actuall label(label means the actual number that we had to guesst), than we see how much each weight and bias
contributed to that error. Here we need partial derivatives!

!one thing, the label data are encoded, here we use the one-hot encoder, this is because we had some choices:

1.Label Encoding
 in this we will just use some number for each possiblility(here we have numbers so we will have done nothink)
 The problem with this is that if we guess wrong, let say the current number we had to guesst was 4, we guesst 6 => the misstake 2, and for the exat example if we had guesst 9 tte mistake = 5
 and the number the ntwork will predist will be 6(lower misstake), but the reality is that we are wrong in both casees!

 2.Binary Encoding
 here for each number we will convert them to binary(0 => 0, 1 => 10, 2 => 100...), but again in the deviation from the correect value we will have the same problem
 
 3.one-hot Encoding
 the right coiche, why
well here each number is represented like:

0 = [0,0,0,0,0,0,0,0,0,0],

1 = [0,1,0,0,0,0,0,0,0,0], 

.....

5 = [0,0,0,0,0,1,0,0,0,0],

so we have a array of 0 and we put 1 in the right index which is equall with the number we ned to represent.

🧮 What’s Actually Happening

At each layer during backprop:

    Compute the gradient of the loss with respect to the output of that layer

    Use the chain rule to break it down into:

        The gradient of the loss w.r.t. the layer’s weights

        The gradient of the loss w.r.t. the layer’s inputs

    Pass that gradient back to the previous layer

This is why it’s called backpropagation of error — we propagate the gradient backward through the network.

Compute derivative of loss w.r.t. Z using the chain rule:

📦 What You Need for Each Layer

    Activation function and its derivative (e.g., ReLU or sigmoid)

    The gradient of the loss (e.g., from softmax + cross-entropy)

    Matrix shapes: careful alignment is critical in NumPy (use .T wisely!)


!!(well this is what ghat-gpt told me -_-, its not too nice. but this is not so hard just go whatch a yt video or reas somewere for this)

And finally Update parameters:
Here the network is optimizing its weights and biases to "learn"

and its pretty strait forward:

W1 = W1 - alpha * dW1

b1 = b1 - alpha * db1  

W2 = W2 - alpha * dW2  

b2 = b2 - alpha * db2  

here alpha is the learning rate this is  called a hyperparameter, it has no connection with any variable in our network(constante).and usually we take this pretty small!

 
And this is the theoritical part, we simply do this in a loop, and the model will "learn", this is too fucking beautifull and elegant.

ps: i have not faild the algebra nor the calculus!!!

And a few more line for the general idea of ML:

Now up till now, we have had a one way relationship with programming. we tell it exatly what to do and it will do it.

But the ML became main stream, and some smart guy introduce another way to programm. 
We are not really writing the code of the thing that you're going to be executing directly.

Essentially you have some sort of a model, that models some sort of a process that you want to predict or do sth with that.

And this model has a certain parameters, and what you code?

You code the description of what you expect from this model, how he should behave.
And then you give you're  untrained model and the description of how its suposed to behave to a learning process, and it just moves the weight and biases around untill it fits you're description well enought.
This with a certant probability, because it will never be perfect.

You can model anything, as long as you have enought data!!

And this is the reason I write "learn", beacause the model its not learning(human prespective). Its just predictiong with a very scary accuracy!



