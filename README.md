# Neural Network Visualisation
Based on Michael Nielsen's explanations at http://neuralnetworksanddeeplearning.com/. A neural network that can be trained to recognise images and a Tkinter visualiser for the trained network

<img src="images/gallery_7.png" title="The window on startup" alt="The number 7 in a pixelated black and white image is displayed to the right.
On the left there are three columns of circles, with the last columns containing the numbers 0 to 9.
The nodes labelled 2 and 7 are in white while the rest are black. The number 7 has been circled" width="400"/>

### The gallery
The gallery view in the top right allowed you to inspect the different training images and what the network predicted.
The colours of the nodes indicates their activity for a given input, white indicating higher activity.
The output node with the highest activity is circled in brown
<img src="images/gallery_2.png" title="Using the gallery feature" alt="The number 2 is displayed to the right.
The node labelled numbers 2 is white and has been circled" width="400"/>

### Navigation
You can zoom and pan through around the network in order to see the nodes better
<img src="images/zoom_1.png" title="Using the zoom feature" alt="The number 1 is displayed to the right.
The node labelled numbers 1 is light grey and has been circled" width="400"/>

There are 784 nodes in the input layer, but only 28 are displayed at a time.
The scrolling arrows allow you to see other nodes that were not initially rendered

<img src="images/gallery_4.png" title="Using the node scrolling feature" alt="The number 4 is displayed to the right.
The nodes labelled numbers 2 and 7 are both dark grey.
The node labelled 2 has been circled" width="400"/>

### Custom input
<img src="images/custom_1.png" title="Using the custom input feature" alt="The number 3 is displayed in the top right corner, in the bottom right
are the handwritten letters spelling out the word 'three'." width="400"/>
<img src="images/custom_2.png" title="Using the custom input feature" alt="A more pixelated version of what was in the bottom right hand corner has moved to the top right.
The node labelled numbers 8 is white and has been circled" width="400"/>

You can draw in the bottom right corner and after pressing test, the image will be evaluated by the network
Alternatively, by setting the ```instant_test``` parameter to be True,
it will automatically input the new image data as you draw
The save button will add the drawn image to a custom image folder than can also be browsed in the gallery
