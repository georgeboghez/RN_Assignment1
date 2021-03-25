import re
import numpy as np


def getValue(coeff):
    if coeff == '-':
        return -1
    if coeff == '' or coeff == '+':
        return 1
    return int(coeff)


def extractCoeff(leftPart, desiredVariable, otherVariableNames):
    variablePart = leftPart.split(desiredVariable)[0].strip()
    if desiredVariable not in leftPart:
        coeff = 0
    elif otherVariableNames[0] in variablePart or otherVariableNames[1] in variablePart:
        coeff = re.split("[{firstVariable}{secondVariable}]".format(
            firstVariable=otherVariableNames[0], secondVariable=otherVariableNames[1]), variablePart)[-1].replace(' ', '')
        coeff = getValue(coeff)
    else:
        coeff = getValue(variablePart)
    return coeff


def parseFile(filename):
    with open(filename, "r") as f:
        content = f.read()
    matrix = []
    results = []
    for equation in content.split('\n'):
        coeffs = []
        eqParts = equation.split("=")
        leftPart = eqParts[0]
        result = int(eqParts[1])
        results.append(result)
        coeffs.append(extractCoeff(leftPart, 'x', 'yz'))
        coeffs.append(extractCoeff(leftPart, 'y', 'xz'))
        coeffs.append(extractCoeff(leftPart, 'z', 'xy'))
        matrix.append(coeffs)
    return [matrix, results]


fileContents = parseFile("file.txt")

A = fileContents[0]
B = fileContents[1]

A_np = []
for row in A:
    A_np.append(np.array(row))
A = np.array(A_np)
print("A:")
print(A)

B = np.array(B)
print("B:")
print(B)

print("Determinant:")
determinant = np.linalg.det(A)
print(determinant)

print("Transpose:")
transpose = A.T
print(transpose)

print("Adjugate:")
# adjugate = np.matrix.getH(A)
adjugate = np.linalg.inv(A) * determinant
print(adjugate)

if int(determinant) == 0:
    print("The matrix isn't invertible, the determinant is 0")
else:
    print("Inverse:")
    inverse = np.linalg.inv(A)
    print(inverse)

    print("Final Results:")
    results = np.matmul(inverse, B)
    print(results)