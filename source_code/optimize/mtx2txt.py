import os


def txt2mtx(filename, new_file_path):
    # print("./sfdp "+filename+" >SFDP/"+str(i)+".txt 2>&1")
    I = []
    J = []
    with open(filename, "r") as file:
        for line in file:
            columns = line.strip().split(" ")
            if len(columns) == 2:
                I.append(int(columns[0]) + 1)
                J.append(int(columns[1]) + 1)
    file.close()

    with open(new_file_path, "w") as file1:
        file1.write("%%MatrixMarket matrix coordinate real symmetric\n")
        unique_elements = set(I + J)
        file1.write(str(len(unique_elements)) + " " + str(len(unique_elements)) + " " + str(len(I)) + "\n")
        # print(X)
        for j in range(len(I)):
            file1.write(str(I[j]) + " " + str(J[j]) + " 1\n")
    file1.close()