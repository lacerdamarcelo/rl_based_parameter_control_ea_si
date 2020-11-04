
This file contains the new instances presented in D.Pisinger
"Where are the hard knapack problems", Computers & Operations
Research (2005). 

Each instance has a name
 
  knapPI_t_n_R.csv

where n is the number of items, R is the range of coefficients, and 
t is the instance type:

   1=ucorrelated
   2=weakly correlated
   3=strongly correlated
   4=inverse strongly correlated
   5=almost strongly correlated
   6=subset sum 
   9=similar weights

   11=uncorrelated span(2,10)
   12=Weakly correlated span(2,10)
   13=Strongly correlated span(2,10)
   14=mstr(3R/10, 2R/10, 6)
   15=pceil(3)
   16=circle(2/3)

There are 100 different instances in each file.
The format of each instance is

  instance name
  n
  c
  z
  time
  1 p[1] w[1] x[1]
  2 p[2] w[2] x[2]
  : 
  n p[n] w[n] x[n]

The solution times (in seconds) have been added to the file to indicate 
the hardness of each instance. The tests have been run on 

  Dell Optiplex 9020 with Intel Core i7-4790 CPU @ 3.60GHz, and 8 GB RAM

using the "combo" knapsack solver. The optimal solution is given in the
x[] vector. Alternative solutions may exist to each instance.

