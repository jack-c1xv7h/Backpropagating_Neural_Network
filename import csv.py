import csv
with open('mnist_train.csv', mode ='r')as file:
  datafile = csv.reader(file)
  for lines in datafile:
        print(lines[0])
        for row in range(0,28):
            rowstring = ""
            for col in range(0,28):
                if lines[row*28+col+1] != "0":
                        rowstring = rowstring + "■ "
                else:
                    rowstring = rowstring + "◻ "
            print(rowstring)
            
        break