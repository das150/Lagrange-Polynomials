import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify
import re

def lagrange_interpolation(x_values, y_values, interval_size=1.0):
	num_intervals = int((max(x_values) - min(x_values)) / interval_size) + 1
	lagrange_polynomials = []

	for i in range(num_intervals):
    	start = min(x_values) + i * interval_size
    	end = start + interval_size
 	 
    	mask = np.logical_and(x_values >= start, x_values <= end)
    	x_interval = x_values[mask]
    	y_interval = y_values[mask]

    	lagrange_poly = lagrange(x_interval, y_interval)
    	lagrange_polynomials.append((lagrange_poly, start, end))
	return lagrange_polynomials

def lagrange(x_values, y_values):
  def lagrange_poly(x):
  	result = 0.0
  	for i in range(len(y_values)):
      	term = y_values[i]
      	for j in range(len(x_values)):
          	if i != j:
              	term *= (x - x_values[j]) / (x_values[i] - x_values[j])
      	result += term
  	return result
  return lagrange_poly

def print_lagrange_equation(lagrange_poly, start, end):
  x = symbols('x')
  lagrange_expr = lagrange_poly(x)
  lagrange_expr_simplified = simplify(lagrange_expr)
  lagrange_str = str(lagrange_expr_simplified).replace('**', '^').replace('*', '')
  lagrange_str = re.sub(r"e(-?\d+)", lambda match: f"(10)^({match.group(1)}))", lagrange_str)
  print(f'Interval [{start:.2f}, {end:.2f}] Equation: r(h) = {lagrange_str}')
 
def plot_lagrange_curves(lagrange_polynomials):
  for lagrange_poly, start, end in lagrange_polynomials:
  	print_lagrange_equation(lagrange_poly, start, end)

  	x_interval = np.linspace(start, end, 1000)
  	y_interval = [lagrange_poly(x) for x in x_interval]
  	plt.plot(x_interval, y_interval, label=f'Interval [{start:.2f}, {end:.2f}]')

  plt.scatter(x_values, y_values, color='red', label='Data Points')

  plt.title('Lagrange Interpolation')
  plt.xlabel('r')
  plt.ylabel('h')
  plt.legend()
  plt.show()
 
def lagrange_equation_str(lagrange_poly, x_interval, start, end):
  terms = []
  for i, x_value in enumerate(x_interval):
  	term = f"{lagrange_poly(x_value):.4f}"
  	for j, other_x in enumerate(x_interval):
      	if j != i:
          	term += f" * (x - {other_x:.2f}) / ({x_value:.2f} - {other_x:.2f})"
  	terms.append(term)
  equation_str = " + ".join(terms)
  return equation_str

x_values = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00,
                 	11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00,
                 	21.00, 21.30])
y_values = np.array([2.55, 2.50, 2.45, 2.50, 2.70, 2.95, 3.30, 3.85, 4.50, 4.90, 5.25, 5.45,
                 	5.70, 5.70, 5.65, 5.60, 5.30, 5.20, 5.00, 4.90, 4.85, 4.70, 4.05])

# Can change
interval_size = 5.0

lagrange_polynomials = lagrange_interpolation(x_values, y_values, interval_size)

plot_lagrange_curves(lagrange_polynomials)
