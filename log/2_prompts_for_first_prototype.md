# Prompts

1. build me a new app here to do automated PID tuning

Built v1, including Zeigler Nichols, AMICO methods

2. does boyd's approach always lead to a solution same or better than ZN AMICO, etc? https://web.stanford.edu/~boyd/papers/pdf/pid_tuning_ecc.pdf

Answered no, and that paper is careful not to claim this.

3. add boyd's method to the mix in pidtuner.py

4. explain open vs close loop ZN method

5. improve pidtuner to model a transfer function of any order, lets say user wants to give input in the form G = 1000 / ((s+1)(10s+1))


