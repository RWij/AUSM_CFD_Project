#/usr/bin/python3

'''
 this is a very hacky way of gathering the coordinates - initialize
 empty arrays and populate them dynamically by appending
 coordinate information. Removes the need to know array size
 info apriori, but is usually bad for big files (~50K+ lines)
'''

# again we use 1D arrays because they are slightly more efficient, but 
# it's easy enough to make them 2D
x = []
y = []

# open file and read line by line to assign to array
with open("g65x49u.dat","r") as fp:
    [nx, ny] = [int(m) for m in fp.readline().strip("\n").split(", ")]
    for line in fp:
        x.append(line.strip("\n").split(", ")[0])
        y.append(line.strip("\n").split(", ")[1])

# note you can also use meshgrid(x,y) to create a mesh out of this!

# write out coordinates again
with open("g65x49u_py.dat","w") as fout:
    fout.write("\t ZONE\t i = %d, j = %d\n" %(nx, ny,))
    for (xx,yy) in zip(x,y):
        fout.write("\t%s,\t%s\n" %(xx,yy,))

# when doing file I/O using "with open ..." you don't need to do file.close()
# python automatically closes the file once the I/O processes are finished
