from os import walk, getcwd
import matplotlib.pyplot as plt

'''
We only need to show every _OTHER_ folder, as each data-set has a 
 training and challenge set. So out of 20 files, we need to show 10

First things first, let us create an array of the directory locations
'''

current_directory = getcwd()
data_sets = "Data-Sets"

path = walk(current_directory + "\\" + data_sets)

directory_array = [] # contains the main folders

i = 1
for root, dirs, files in path:
    if (i == 2):
        directory_array = dirs
        break
    
    i += 1

print(directory_array)

print("\nStarting matplotlib\n")

path = walk(current_directory + "\\" + data_sets) # reset path

fig = plt.figure(figsize=(14, 8))

i = 0
for root, dirs, files in path:
    # print(dirs)
    for item in files:
        # only execute for first picture in directory
        if ("t0000.tif" == item) or ("t000.tif" == item):
            i += 1

            # saftey, incase i gets too large
            if (i > 19):
                break
            elif (i % 2 == 0):
                continue
            print(item)
            location = ( current_directory + "\\" + data_sets + "\\Extracted\\" + directory_array[i] + 
                        "\\" + directory_array[i] + "\\01\\" + item)
            print(location)

            # img = plt.imread(location)
            
            # fig.add_subplot(2, 5, i+1)
            # plt.title(item)
            # plt.imshow(img, cmap="gray")

            
            break

        else:
            break

plt.tight_layout()
plt.show()
