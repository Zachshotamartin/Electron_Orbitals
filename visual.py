import numpy as np
import cv2
import sympy as sp
import math
from scipy.interpolate import interp1d

a = (0.529*10**-10)
e = np.exp(1)
n = 13
l = 7
m = 2

#Class for Computing Nth Derivatives and Associated Leguerre+Legendre Polynomials

class NthDifferentiation:
    def __init__(self, function):
        self.function = function
        self.x = sp.symbols('x')
        self.f = sp.sympify(function)

    def nth_derivative(self, n):
        return sp.diff(self.f, self.x, n)

# Calculate only once
def AssociatedLeguerrePolynomial(p, q):
    #Define global variable x
    x = sp.symbols('x')
    # Create the first function
    func1_string = f'exp(-x)*(x**{p + q})'
    func1 = NthDifferentiation(func1_string)
    A = func1.nth_derivative(p + q)
    A_simplified = sp.simplify(A)
    func2_expr = sp.exp(x) * A_simplified
    func2 = NthDifferentiation(func2_expr)
    B = func2.nth_derivative(p)
    result = (((-1)**p) * (1/math.factorial(p+q)) * B)
    func = sp.lambdify(x, result, modules= 'numpy')
    return(func)

# Calculate only once
def AssociatedLegendrePolynomial(m, l):
    #Define global variable x
    x = sp.symbols('x')
    #Create the first function
    func1_string = f'(x**2 - 1)**{l}'
    func1 = NthDifferentiation(func1_string)
    A = func1.nth_derivative(l)
   
    result = A * (1 / ((2**l)*(math.factorial(l)) ))
       
    A_simplified = sp.simplify(result)
    func2 =  NthDifferentiation(A_simplified)
    B = func2.nth_derivative(m)
   
    result =  ((1 - x**2)**(m/2)) * B
    func = sp.lambdify(x, result, modules='numpy')
    return(func)

# Depends on n, l, r, and the associated leguerre polynomial
def Prob_Angular(n, l, r, L):
    M = r*(2/(n*a))
    Term1 = (np.sqrt( ((2/(n*a)**3)) * (math.factorial(n-l-1) / (2*n*math.factorial(n+l)))))
    Term2 = (np.exp(-0.5*M))*((M)**l)
    Radial = Term1 * Term2 * L(M)
    return (Radial)
    
    #Angular Equation

# Depends on l, m, theta, and the associated legendre polynomial
def Prob_Radial(l, m, theta, T):
    x = np.cos(theta)   
    Term1 = np.sqrt( ((2*l+1)*(math.factorial(l-m))) / ((4*np.pi)*(math.factorial(l+m))) )
    Angular = T(x)*Term1
    return(Angular)


class visual: 
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.z = np.zeros((height, width), dtype=np.float32)
    def generate(self):
        T = AssociatedLegendrePolynomial(m, l)
        L = AssociatedLeguerrePolynomial(2 * l +1, n - l -1)
        for y in range(self.height):
            if (y % 20 == 0):
                print(f"Processing row ({y})")
            for x in range(self.width):
                radius, angle = self.get_radius_and_angle(x, y)
                
                radius = np.interp(radius,[0,np.sqrt((self.height//2)**2 + (self.width//2)**2)],[0,28 * 10**-9])
                AWF = Prob_Angular(n, l, radius, L)
                RWF = Prob_Radial(l, m, angle, T)
                Z = (AWF * RWF)**2
                self.z[y, x] = Z
        max_z = np.max(self.z)
        min_z = np.min(self.z)
        self.screen[:, :, 0] = np.interp(self.z, [min_z, max_z], [0, 255]).astype(np.uint8)  # Adjust color mapping for red channel
        self.screen[:, :, 1] = np.interp(self.z, [min_z, max_z], [0, 255]).astype(np.uint8)  # Adjust color mapping for green channel
        self.screen[:, :, 2] = np.interp(self.z, [min_z, max_z], [0, 255]).astype(np.uint8)  # Adjust color mapping for blue channel
        self.screen = cv2.applyColorMap(self.screen, cv2.COLORMAP_INFERNO)
        

    def get_radius_and_angle(self, x, y):
        radius = np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)
        angle = np.arctan2(self.center_y - y, x - self.center_x)
        if angle < 0:
            angle += 2 * np.pi
        return radius, angle

def main():

    image = visual(7680, 4320)
    image.generate()
    # Change to your desired path !!!
    cv2.imwrite("/Users/zacharymartin/Desktop/jacob/output.png", image.screen)
    
if __name__ == '__main__':
    main()
