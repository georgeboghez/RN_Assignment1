import re


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


def getDeterminant(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        return matrix[0][0] * matrix[1][1] * matrix[2][2] + matrix[1][0] * matrix[2][1] * matrix[0][2] + matrix[0][1] * matrix[1][2] * matrix[2][0] - (matrix[0][2] * matrix[1][1] * matrix[2][0] + matrix[0][0] * matrix[1][2] * matrix[2][1] + matrix[2][2] * matrix[0][1] * matrix[1][0])


def getTranspose(matrix):
    transpose = [[], [], []]
    for row in matrix:
        transpose[0].append(row[0])
        transpose[1].append(row[1])
        transpose[2].append(row[2])
    return transpose


def getAdjugate(transpose):
    adjugate = []
    for i in range(len(transpose)):
        adjugateRow = []
        for j in range(len(transpose[i])):
            newMatrix = []
            for ii in range(len(transpose)):
                newRow = []
                for jj in range(len(transpose[ii])):
                    if ii != i and jj != j:
                        newRow.append(transpose[ii][jj])
                if newRow:
                    newMatrix.append(newRow)
            adjugateRow.append(pow(-1, ((i + 1) + (j + 1)))
                               * getDeterminant(newMatrix))
        adjugate.append(adjugateRow)
    return adjugate


def getInverse(adjugate, determinant):
    for i in range(len(adjugate)):
        for j in range(len(adjugate[i])):
            adjugate[i][j] = 1 / determinant * adjugate[i][j]
    return adjugate


def finalResult(inverse, results):
    finalResults = []
    for i in range(len(inverse)):
        result = 0
        for j in range(len(inverse[i])):
            result += inverse[i][j] * results[j]
        finalResults.append(result)
    return finalResults


fileContents = parseFile("file.txt")

A = fileContents[0]
B = fileContents[1]

print("A:")
print(A)

print("B:")
print(B)

determinant = getDeterminant(A)
print("Determinant:")
print(determinant)

transpose = getTranspose(A)
print("Transpose:")
print(transpose)

adjugate = getAdjugate(transpose)
print("Adjugate:")
print(adjugate)

inverse = getInverse(adjugate, determinant)
print("Inverse:")
print(inverse)

X = finalResult(inverse, B)
print("Final Results:")
print(X)
