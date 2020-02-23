# Fuzzy-RB-System
Simple fuzzy rule-based logic inference system based on the Mamdani principles.


The system can provide inference from scratch or using <b>[scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)</b>'s control system.  

The fuzzy variables are represented as 4-tuples with trapezoidal membership functions. The defuzzification methods used are <b>bisector</b> and <b>centroid</b>.

### Example rulebase for defuzzification

```txt
anestheticsRulebase

Rule 1: If HR is normal and R is normal then D is average
Rule 2: If HR is low and R is normal then D is moderate
Rule 3: If HR is low and R is low then D is small
Rule 4: If HR is very_low and R is low then D is very_small
Rule 5: If HR is high and R is high then D is large
Rule 6: If HR is very_high and R is high then D is very_large

HR

very_low 40 40 0 20
low 60 60 20 10
normal 70 90 10 10
high 100 100 10 20
very_high 120 120 20 0

R

low 0 3 0 3
normal 6 8 3 2
high 10 12 2 0

D

very_small 0 0 0 2
small 2 2 2 2
moderate 4 4 2 2
average 6 6 2 2
large 8 8 2 2
very_large 10 10 2 2

HR = 55
R = 4
```

![Defuzz](https://github.com/JadeBlue96/Fuzzy-RB-System/blob/master/defuzz_sc.PNG)

### Instructions for running the examples

In order to install the libraries, the recommended method (with Anaconda), is to create a new environment and install the versions specified in the requirements file:
```python
conda create -n fuzzy python=3.6
conda activate fuzzy

# then in the project directory
pip install -r requirements.txt
```
The basic examples can be run from the command line by passing the input knowledge base as an argument using the following:
```python
# from scratch
python fuzzy_test_main.py anesthetics.fuzzy 
# or using scikit-fuzzy
python fuzzy_test_control.py anesthetics.fuzzy
```

For both examples, a visualization of the current fuzzy set will pop up and the computation will continue once the figure window is closed.


Regarding the simulations performed in the testing section, they can also be run in a similar fashion:
```python
#example: python simulate_measurements.py anesthetics.fuzzy HR R 10
python simulate_measurements.py <fuzzy_filename> <antc_i> <antc_j> <step_size>

#example: python simulate_fuzzy_sets.py anesthetics.fuzzy D 200
python simulate_fuzzy_sets.py <fuzzy_filename> <conseq_var> <step_size>
```
