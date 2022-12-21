# meltingpot
You will find two scripts in this repository, the first one 'nn_multiplication.py' performs number multiplication 
and the second one 'seq2seq_reverse_sequence.py' takes a sequence of numbers as input and returns it reverse sequence.

#Quick Start
## Multiplication
Takes as input a sequence of two numbers specified by the parameter -l (details in the script) 
models can be: fnn, rnn, lstm or bilstm

Commandline:
          python3 nn_multiplication.py --model lstm -l 9 2  

Input
[9. 2.]\\
Predicted multiplication output\\
[18.04326]

## Reverse sequence

Takes as input a sequence of numbers specified by the parameter -l (details in the script) 
models can be:  rnn, lstm or bilstm

python3 seq2seq_reverse_sequence.py --model lstm -l 8 2 3 6

Result:

Input 
[8.0, 2.0, 3.0, 6.0]
Predicted output
[5.83, 2.7, 1.82, 7.51]
