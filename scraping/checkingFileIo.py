import os

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'test.txt')
print(my_file)
file = open(my_file,'x')
file = open(my_file,'w')
file.write('whatever')
file.close()