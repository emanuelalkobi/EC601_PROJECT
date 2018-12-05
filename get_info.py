from os import access, R_OK
from os.path import isfile

def get_video_file(filename):
    assert isfile(filename) and access(filename, R_OK),"File {} doesn't exist or isn't readable".format(filename)
    return(filename)

def get_name1():
    team_name = input('Name team number 1 (attacking towards up or left of the screen)\n')
    return team_name

def get_name2():
    team_name = input('Name team number 2 (attacking towards down or right of the screen)\n')
    return team_name

def get_color(i,colors):
    color1 = None
    while color1 is None:
        try:
            color1 = int(input('Choose team number'+str(i)+' color(enter a number):\n1.red\n2.yellow\n3.green\n4.blue\n '))
        except ValueError:
            print("Program halted incorrect data entered,please Enter a number ")
    while (color1<0 and color1> len(colors)):
         print("you insert an unsupported color please insert another color from the list\n")
         color1 = input('Choose team number'+str(i)+' color(enter a number):\n1.red\n2.yellow\n3.green\n4.blue\n ')
    color1=int(color1)
    #python array starts from 0
    return(colors[color1-1])

def get_line_direction():
    line_dir = input('Please choose the offside line:1.horizontal 2.vertical 3.no line\n')
    line_dir = int(line_dir)
    return line_dir

def get_game_direction():
    game_direction = input('Please choose the game direction: 1.horizontal 2.vertical\n')
    game_direction = int(game_direction)
    return game_direction


