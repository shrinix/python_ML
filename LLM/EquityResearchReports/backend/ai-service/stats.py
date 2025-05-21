import math
from decimal import Decimal, getcontext
import numpy as np
# Set precision for Decimal calculations
getcontext().prec = 28

# function to calculate mean using sliding window and previous mean
# old_value is the value that is being removed from the window
def inc_mean(mean, new_value, old_value, n):
    """
    Incremental mean calculation.
    
    Parameters:
    mean (float): Previous mean.
    new_value (float): New value to include in the calculation.
    old_value (float): Old value to exclude from the calculation.
    n (int): Number of values in the current window.
    
    Returns:
    return Decimal(mean) + (Decimal(new_value) - Decimal(old_value)) / Decimal(n)
    """
    if n <= 1:
        return new_value
    return mean + (new_value - old_value) / n

def my_inc_sd(prev_sd, prev_mean, new_value, old_value, n):

    new_mean = Decimal(prev_mean) + (Decimal(new_value) - Decimal(old_value)) / Decimal(n)
    first_term = Decimal(n - 2) * (Decimal(prev_sd) ** 2)
    second_term = (Decimal(new_value) - Decimal(new_mean)) * (Decimal(new_value) - Decimal(prev_mean))
    
    sd = Decimal(np.sqrt((first_term + second_term) / Decimal(n - 1)))
    return sd

def inc_sd(prev_sd, prev_mean, new_value, old_value, n):
    """
    Incremental standard deviation calculation using the correct formula.

    Parameters:
        prev_sd (float): Previous standard deviation.
        prev_mean (float): Previous mean.
        new_value (float): New value to include in the calculation.
        old_value (float): Old value to exclude from the calculation.
        n (int): Window size.

    Returns:
        float: Updated standard deviation.
    """
    prev_variance = Decimal(prev_sd) ** 2

    # Calculate previous variance
    prev_variance = prev_sd ** 2

    new_variance = (
        Decimal(prev_variance)
        + ((Decimal(new_value) - Decimal(prev_mean)) ** 2 - (Decimal(old_value) - Decimal(prev_mean)) ** 2) / Decimal(n)
        - ((Decimal(new_value) - Decimal(old_value)) * (Decimal(new_mean) - Decimal(prev_mean))) / Decimal(n)
    )
    new_variance = max(
        Decimal(prev_variance)
        + ((Decimal(new_value) - Decimal(prev_mean)) ** 2 - (Decimal(old_value) - Decimal(prev_mean)) ** 2) / Decimal(n)
        - ((Decimal(new_value) - Decimal(old_value)) * (Decimal(new_mean) - Decimal(prev_mean))) / Decimal(n),
        Decimal(0)
    )

    new_sd = Decimal(new_variance).sqrt()
    new_variance = max(new_variance, 0)

    # Calculate new standard deviation
    new_sd = new_variance ** 0.5

    return new_sd

data_stream = [240.360001, 237.300003, 241.839996, 238.029999, 235.929993, 235.740005, 235.330002,
            239.070007, 227.479996, 220.839996, 216.979996, 209.679993, 213.490005, 214, 212.690002,
            215.240005, 214.100006, 218.270004, 220.729996, 223.75, 221.529999]

#Example 1
n=20 #window_size
old_value_date='02/26/25' #the date which is just outside the 20 day window
new_value_date='03/26/25'
prev_mean_and_sd_date='03/25/25' #one day before the current date
prev_mean = 225.5425 #mean value on the previous day 03/25/25
prev_sd = 10.910631 #SD value on the previous day 03/25/25
old_value = 240.360001 #popout_value Value being removed from the window i.e. the value of ADJ Close on 02/26/25
new_value = 221.529999 #New value being added to the window i.e. the value of ADJ Close on 03/26/25
 
new_mean = inc_mean(prev_mean, new_value, old_value, n)
print(f"New Mean: {new_mean}")
new_sd = my_inc_sd(prev_sd, prev_mean, new_value, old_value, n)
print(f"New SD: {new_sd}")

#Example 2
# n=20 #window_size
old_value_date='02/24/25'  #the date which is just outside the 20 day window
new_value_date='03/25/25'
prev_mean_and_sd_date='03/24/25' #one day before the current date
prev_mean = 226.707 #mean value on the previous day 03/24/25
prev_sd = 11.858847 #SD value on the previous day 03/24/25
old_value = 247.039993 #popout_value Value being removed from the window i.e. the value of ADJ Close on 02/24/25
new_value = 223.75 #New value being added to the window i.e. the value of ADJ Close on 03/25/25

new_mean = inc_mean(prev_mean, new_value, old_value, n)
print(f"New Mean: {new_mean}")
new_sd = my_inc_sd(prev_sd, prev_mean, new_value, old_value, n)
print(f"New SD: {new_sd}")