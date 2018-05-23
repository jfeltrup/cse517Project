# A file to create small datasets from out larger data to test the model

def main():
    print("Program start")
    sentences = processData()

    # Print the sentences to a new files
    f = open("small_dataset_100.txt", "w");
    for sentence in sentences:
        f.write(sentence)

    print("Program Finished")


# Reads in the training Data from a file, and converts it into a list of unicode sentences
# This list of sentences is returned
def processData():
    f = open("DryRunModel/dataset_april.txt", encoding="utf-8")
    sentences = []
    line = f.readline()
    # Use count when you want to limit the number of lines read, for time
    count = 0
    while line:
        sentences.append(line)
        line = f.readline()
        count += 1;
        if count == 100:
            break;
    return sentences


if __name__ == '__main__':
    main()