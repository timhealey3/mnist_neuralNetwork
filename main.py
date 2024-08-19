import mnist_nn as nn

def displayText():
    print("N. Simple Neural Network")
    print("C. Convuluted Neural Network")
    print("Q. Quit program\n")

def getInput():
    return input(">> ")

programRunning = True
while programRunning:
    displayText()
    userInput = getInput()

    match userInput:
        case "n":
            print("neural network has been selected")
            nn.createModel()
            nn.runModel()
        case "c":
            print("CNN has been selected")
        case "q":
            programRunning = False
        case _:
            print("Please select either C, N, or Q")
