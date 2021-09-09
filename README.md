# English-tamil-transliterator <br>
This is a program is used to translitrate english words to tamil words <br>
TECH stack used:<br>
* Deep learning (attention based Recurrent neural network) <br>
Input to the program:
English words <br>
Output to the program:
Tamil words <br>
Algorithm: <br>
Get the input from user 
Preprocess the input (Make it as words)
Make it as compatible tensor <br>
Pass that Tensor to the Model <br>
Get the output from the model <br>
Post process the output to get in the tamil words (i.e model will produce the output tensor and convert these tensor into tamil sentence) <br>
Model: <br>
Two layer bidirectional LSTM used as encoder 
