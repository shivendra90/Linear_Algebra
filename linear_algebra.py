"""Collection of linear algebra methods.

Created on Fri Mar  8 13:01:35 2019
"""
import numpy as np
from numpy import array
import math
__author__ = "Shivendra"
__Version__ = '1.0'


class Vector:
    """Simple vector operations in Python."""

    def __init__(self, coordinates):
        """Initialize 2D/1D vectors."""
        self.coordinates = coordinates
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.length = len(coordinates)

        except TypeError:
            raise TypeError("Incorrect type of input")

        except ValueError:
            raise ValueError("Incorrect coordinate values")

    def add(self, other):
        """Execute addition operation."""
        new_values = [x + y for x, y in zip(self.coordinates, other.coordinates)]
        return Vector(new_values)

    def subtract(self, other):
        """Execute subtraction."""
        new_values = [x - y for x, y in zip(self.coordinates, other.coordinates)]
        return Vector(new_values)

    def scalar(self, a):
        """Execute typical matrix scalar operation."""
        new_values = [round(a * x, 3) for x in self.coordinates]
        return new_values

    def magnitude(self):
        """Calculate magnitude of a vector."""
        summation = 0
        for number in self.coordinates:
            summation += round(number ** 2, 3)
        return round(math.sqrt(summation), 3)

    def direction(self):  # Also called normalization
        """Find out direction of vector."""
        try:
            mag = self.magnitude()
            new_values = self.scalar(1. / mag)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector(new_values)

    def dot_product(self, x):
        """Calculate dot product of two vectors."""
        new_value = sum(
            [i * j for i, j in zip(self.coordinates, x.coordinates)])
        return round(new_value, 3)

    def cosine(self, x, in_degrees=False):
        """Determine cosine angle between two vectors."""
        try:
            u_1 = self.direction()
            u_2 = x.direction()

            angle_in_radians = math.acos(u_1.dot_product(u_2))

            if in_degrees:
                degree_per_radian = 180. / math.pi
                return round(angle_in_radians * degree_per_radian, 2)
            return round(angle_in_radians, 2)
        except Exception as e:
            if str(e) == "Cannot divide with zero vector":
                raise Exception("Cannot compute angle with zero vector")
            raise e

    def sine(self, x, in_degrees=False):
        """Calculate sine angle between two vectors."""
        try:
            u_1 = self.direction()
            u_2 = x.direction()

            angle_in_radians = math.asin(u_1.dot_product(u_2))

            if in_degrees:
                degree_per_radian = 180. / math.pi
                return round(angle_in_radians * degree_per_radian, 2)
            else:
                return round(angle_in_radians, 2)
        except Exception as e:
            if str(e) == "Cannot divide with zero vector":
                raise Exception("Cannot compute angle with zero vector")
            raise e

    def is_parallel(self, x):
        """Determine whether two vectors are parallel."""
        u_1 = 0
        u_2 = 0
        if x:
            for i in self.coordinates:
                u_1 += abs(i)
            for j in x.coordinates:
                u_2 += abs(j)
        if u_2 != 0:
            if int(u_2) % int(u_1) < 1.2:
                return "Vectors are parallel."
            else:
                return "Vectors are not parallel."
        else:
            return "Vectors are parallel."

    def is_orthogonal(self, x):
        """Determine if two vectors are perpendicular/orthogonal."""
        if not x:
            raise Exception("Cannot compute with a missing vector.")
        dot = int(self.dot_product(x))

        if dot == 0:
            return "Vectors are orthogonal."
        return "Vectors are not orthogonal."

    def projection(self, x):
        """
        Calculate projection of a vector onto x.

        Formula: x(dot(self, x)/(|x|^2))
        """
        self_dot_x = sum([i * j for i, j in zip(self.coordinates, x.coordinates)])

        x_len = sum([i * i for i in x.coordinates])
        temp = self_dot_x / x_len
        comp = [round(i * temp, 3) for i in x.coordinates]
        return comp

    def component(self, x):
        """
        Calculate component of a vector.

        Formula: Fx = F.cos_theta, Fy = F.sine_theta
        """
        v_self = self.magnitude()
        v_x = round(v_self * self.cosine(x), 3)
        v_y = round(v_self * self.sine(x), 3)
        return [v_x, v_y]

    def cross_product(self, other, is_3d=True):
        """
        Calculate cross product of two matrices.

        Assuming matrices are 3d, return the cross product.
        """
        # Fill in each element
        values = [round((self.coordinates[1] * other.coordinates[2] - other.coordinates[1] * self.coordinates[2]), 3),
                  round(-(self.coordinates[0] * other.coordinates[2] -
                          other.coordinates[0] * self.coordinates[2]), 3),
                  round((self.coordinates[0] * other.coordinates[1] - other.coordinates[0] * self.coordinates[1]), 3)]
        if is_3d:
            return values
        else:
            raise Exception("Cannot return cross product for 2d vectors.")

    def area_parallelogram(self, other):
        """
        Calculate area of parallelogram.

        Formula: (||self|| * ||other||)sine_theta
        """
        cross = self.cross_product(other)
        area = math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
        return round(area, 3)

    def area_triangle(self, other):
        """Return area of triangle given two matrices."""
        cross = self.cross_product(other)
        area = (math.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) / 2)
        return round(area, 3)

    def __str__(self):
        """Print vector method."""
        return "Vector:: {}".format(self.coordinates, '.3f')

    def __eq__(self, x):
        """Determine whether two vectors are the same."""
        return self.coordinates == x.coordinates

    def __truediv__(self, other):
        """Return true division of two vectors."""
        return self / other

    def __len__(self):
        """Returns length of vector."""
        return self.length


class Line:
    """
    Algebraic implementation for lines and linear equations.

    The most common form of expression for linear equations m = Ax + By,
    where m is the intercept of the line and the slope is the ratio
    of coefficients of y and x (B/A). B and A are also called as beta and
    alpha variables especially in statistical applications.
    """

    def __init__(self, x_coefficient=0.0, y_coefficient=0.0, intercept=0.0, normal_vector=None):
        """Initialize 2D lines."""
        self.dimension = 2
        self.x_coefficient = x_coefficient
        self.y_coefficient = y_coefficient
        self.intercept = intercept

        if not normal_vector:
            all_zeros = [0] * self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

    def is_identical(self, other) -> str:
        """Determines if two lines are equal/overlapping."""
        try:
            x_1 = (self.intercept - self.y_coefficient) / self.x_coefficient
            x_2 = (other.intercept - other.y_coefficient) / other.x_coefficient

            y_1 = (self.intercept - self.x_coefficient) / self.y_coefficient
            y_2 = (other.intercept - other.x_coefficient) / other.y_coefficient

            if round(x_1, 1) == round(x_2, 1) and round(y_1, 1) == round(y_2, 1):
                return f"Congruency pairs {round(x_1, 1), round(y_1, 1)} and {round(x_2, 1), round(y_2, 1)}."
            else:
                return f"X-coordinates: {round(x_1, 1)} and {round(x_2, 1)}."

        except TypeError:
            return "Cannot multiply numbers with alphabets."
        except ValueError:
            return "Invalid values entered."

    def is_parallel(self, other):
        """Returns whether two lines are parallel.

        Lines can be parallel only if their slopes are equal.
        """
        if self.x_coefficient == other.x_coefficient and self.y_coefficient == other.y_coefficient:
            return "Lines are parallel because of identical slopes."
        else:
            return "Lines are not parallel."

    def is_intersecting(self, other):
        """Determines if two lines intersect."""
        if self.x_coefficient == other.x_coefficient and self.y_coefficient == other.y_coefficient:
            return "Lines are parallel and hence cannot intersect."
        elif self.intercept == other.intercept and self.x_coefficient == -other.x_coefficient:
            return "Lines are orthogonal."
        elif self.y_coefficient == 0 and other.x_coefficient == 0 and self.intercept == other.intercept:  # 90 angle
            return "Lines are orthogonal with one being vertical and other being horizontal with common intercepts."
        else:  # Solve for x and y
            x = self.intercept / self.x_coefficient
            y = other.intercept / other.y_coefficient
            return f"Lines intersect at {round(x, 3), round(y, 3)}."

    def __str__(self):  # Represented as Ax + By = M
        """Print method for 2D vectors."""
        return "Equation:: {}x + {}y = {}".format(self.x_coefficient, self.y_coefficient, round(self.intercept, 3))


class Plane:
    """Class for a 3 dimensional plane.

    Basically the same Line class with more dimensions.
    """

    def __init__(self, x_coefficient=0.0, y_coefficient=0.0, z_coefficient=0.0, intercept=0.0, normal_vector=None):
        """Initialize 3D planes."""
        self.dimension = 3
        self.x_coefficient = x_coefficient
        self.y_coefficient = y_coefficient
        self.z_coefficient = z_coefficient
        self.intercept = intercept
        self.normal_vector = normal_vector

        if not normal_vector:
            all_zeros = [0] * self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

    def is_parallel(self, other):  # Works similar to line parallelism
        """Checks if two planes are parallel."""
        if self.x_coefficient == other.x_coefficient and self.y_coefficient == other.y_coefficient and \
                self.z_coefficient == other.z_coefficient:
            return f"Coordinates for parallelism: {round(self.x_coefficient, 0), round(self.y_coefficient, 0), round(self.z_coefficient, 0)}" \
                f"and {round(other.x_coefficient, 0), round(other.y_coefficient, 0), round(other.z_coefficient, 0)}."
        else:
            return f"Planes are not parallel: {round(self.x_coefficient, 0), round(self.y_coefficient, 0), round(self.z_coefficient, 0)}" \
                  f" and {round(other.x_coefficient, 0), round(other.y_coefficient, 0), round(other.z_coefficient, 0)}."

    def is_identical(self, other):  # Works similar to identical lines
        """Determine if two planes are identical."""
        x_1 = round((self.intercept - self.y_coefficient -
                    self.z_coefficient) / self.x_coefficient, 0)
        x_2 = round((other.intercept - other.y_coefficient -
                    other.z_coefficient) / other.x_coefficient, 0)

        y_1 = round((self.intercept - self.x_coefficient -
                    self.z_coefficient) / self.y_coefficient, 0)
        y_2 = round((other.intercept - other.x_coefficient -
                    other.z_coefficient) / other.y_coefficient, 0)

        z_1 = round((self.intercept - self.x_coefficient -
                    self.y_coefficient) / self.z_coefficient, 0)
        z_2 = round((other.intercept - other.x_coefficient -
                    other.y_coefficient) / other.z_coefficient, 0)
        # Note: Can also be solved through matrix division. If found a common factor, then planes are identical.

        if x_1 == x_2 and y_1 == y_2 and z_1 == z_2:
            return f"Planes are identical: {x_1, y_1, z_1} and {x_2, y_2, z_2}."
        else:
            return f"Planes are non-identical: {x_1, y_1, z_1} and {x_2, y_2, z_2}."

    def __str__(self):  # Represented as Ax + By + Cz = M
        """Print method for planes."""
        return "Equation:: {}x + {}y + {}z = {}".format(self.x_coefficient,
                                                        self.y_coefficient,
                                                        self.z_coefficient,
                                                        round(self.intercept, 3))


class LinearSystem:
    """Class for linear system."""

    all_planes_in_same_dim = "All planes should be in the same dimension."
    no_solutions_msg = "No solutions."
    inf_solutions_msg = "Infinite solutions."
    zero_pivots_msg = "Zero diagonal elements; system has no solution."

    def __init__(self, unknowns):
        """Linear system initializer."""
        try:
            self.unknowns = list(unknowns)
            self.rows = len(self.unknowns)
            self.cols = len(self.unknowns[0])

            if not unknowns:
                raise ValueError("Invalid or no inputs entered.")
        except TypeError:
            raise Exception("Invalid input type.")

    def matrix_formation(self):
        """Forms a m by n matrix where m is number of rows
        and n is number of columns."""
        mat = array([self.unknowns])
        return mat

    def gauss_jordan(self, b=[1, 0, 0]):  # Elimination process
        """
        Performs standard Gauss-Jordan elimination process.
        A typical 3D augmented matrix is denoted as below:
        [a11 a12 a13 | a14]
        [a21 a22 a23 | a24]
        [a31 a32 a33 | a34]
        where a11...a33 are all coefficients of a linear system.
        The Gaussian process provides the triangular form of the
        above matrix:
        [a11 a12 a13 | a14]
        [0   a22 a23 | a23]
        [0   0   a33 | a33]
        The Gauss-Jordan process takes things further and performs
        the reverse elimination process to provide with the identity
        matrix:
        [1 0 0 | a14]
        [0 1 0 | a24]
        [0 0 1 | a34]
        """
        a = array(self.unknowns, dtype='float')
        b = array(b, dtype='float')
        Ab = np.hstack([a, b.reshape(-1, 1)])

        m = len(Ab)
        n = len(Ab[0])

        Ab = list(Ab)

        for row, col in enumerate(Ab):
            if col[row] != 0:
                divisor = col[row]
            else:
                raise ValueError(self.zero_pivots_msg)

            for ind, term in enumerate(col):
                col[ind] = term / divisor

            for i in range(m):
                if i != row:
                    inv = -1 * Ab[i][row]
                    for j in range(n):
                        Ab[i][j] += inv * Ab[row][j]

        if self.rows < self.cols or self.rows > self.cols:
            print("System has one or more free variables.")
        else:
            print("System has a unique solution.")
        
        out = array(Ab)
        return print(out)

    def __str__(self):
        mat = array(self.unknowns)
        return str(mat)

    def __len__(self):
        """Returns dimensions of the matrix."""
        rows, columns = self.rows, self.cols
        return f"{rows} by {columns} array."


# Example
vec_1 = LinearSystem([[2, -1, 5, 1], [3, 2, 2, -6],
                      [1, 3, 3, -1], [5, -2, -3, 3]])
vec_1.gauss_jordan([-3, -32, -47, 49])
