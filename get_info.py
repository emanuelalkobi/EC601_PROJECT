from os import access, R_OK
from os.path import isfile

def get_video_file(filename):
    assert isfile(filename) and access(filename, R_OK),"File {} doesn't exist or isn't readable".format(filename)
    return(filename)
def get_name(i):
    team_name = input('Choose team number'+str(i)+' name :\n ')
    return team_name

def get_color(i,colors):
    print("get_color!!!!",colors)
    color1=-1
    color1 = input('Choose team number'+str(i)+' color(enter a number):\n1.red\n2.yellow\n3.green\n4.blue\n ')
    color1=int(color1)
    while (color1<0 and color1> len(colors)):
         print("you insert an unsupported color please insert another color from the list\n")
         color1 = input('Choose team number'+str(i)+' color(enter a number):\n1.red\n2.yellow\n3.green\n4.blue\n ')
    color1=int(color1)
    #python array starts from 0
    return(colors[color1-1])
