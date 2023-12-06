# Data-Availability

## Steps to reproduce data

- Download or clone the git project to your local computer (envirenment)
- Make sure you have installed and available python3 on your computer (envirenment)
- Install all the python libraries that are used in fitting-data.py file if needed
- In cal_settings.py file we have initial values for 33 datasets that are numerated and correspond to the numbers in the Table 1 of SI
- To fit the data for the given number, in cal_settings.py file uncomment only the initial values for corresponding number that is mentioned in comments 
- Then run fitting-data.py file with python3, and you will see the plotted graph and the fitting results will be printed on the terminal
- After you know the fitting results you can compare it with the corresponding values in output.txt