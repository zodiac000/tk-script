from tkinter import *
from PIL import ImageTk, Image
import os
import re
from collections import OrderedDict
import pyscreenshot as ImageGrab
import pdb


class ImageLabel:
    def __init__(self):
        self.pointer = 0
        self.dicts = {}
        self.xys = []
        self.directs = []
        self.root = Tk()
        self.prevOval = None
        self.current_image = None
        self.only_invalid = False

        self.top_10_pointer = 0
        self.top_10 = [262, 828, 842, 842, 545, 328, 747, 846, 324, 326]
        self.saved_dict = "csv/saved_dict.csv"
        self.coordinates = "csv/coordinates.txt"
        # self.pred_dict = "csv/pred_dict_4265.csv"
        # self.pred_dict = "csv/pred_dict_200.csv"
        # self.pred_dict = "csv/pred_dict_200_G.csv"
        # self.pred_dict = "csv/pred_dict_cascade_4265.csv"
        self.pred_dict = "csv/pred_dict_cascade4_6500.csv"
        self.pass_csv = 'csv/pass_images.csv'

    def callback_save_point(self, event):
        msg = "clicked at " + str(event.x) + " " + str(event.y) + "\n"
        print(msg)
        with open("out","ab") as f:
            f.write(msg.encode())

    def read_xys_from_file(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
        coordinates = []
        for line in lines:
            coordinate = line.split(",")
            coordinates.append((int(coordinate[0]), int(coordinate[1])))
        return coordinates

    def load_dictionary_from_osdir(self, image_loc, xy_loc):
        images = os.listdir(image_loc)
        images = list(filter(lambda x: re.search("\.jpg$", x), images)) 
        xys = self.read_xys_from_file(xy_loc)
        lens = len(images) if len(images) < len(xys) else len(xys)
        print("{} images has been loaded".format(str(lens)))
        dicts = {}
        for i in range(lens):
            dicts[os.path.join(image_loc, images[i])] = xys[i]
        self.dicts = OrderedDict(sorted(dicts.items(), key=lambda t: t[0]))
        self.load_directs_xys()

    def load_dictionary_from_dict(self, dict_loc):
        with open(dict_loc, 'r') as file:
            lines = file.readlines()
        self.dicts = {}
        for line in lines:
            splited = line.split(",")
            self.dicts[splited[0]] = (int(splited[1]), int(splited[2]))
        self.load_directs_xys()
            

    def callback_next(self, event):
        if self.pointer < len(self.directs) - 1:
            canvas.delete("all") 
            self.pointer += 1
            if self.only_invalid:
                while(self.xys[self.pointer][0] >= 0):
                    self.pointer += 1
            self.load_image()
            canvas.create_image(0, 0, image=self.current_image, anchor=NW)
            self.auto_paint()
            self.prevOval = None

    def callback_previous(self, event):
        if self.pointer > 0:
            canvas.delete("all") 
            self.pointer -= 1
            if self.only_invalid:
                while(self.xys[self.pointer][0] >= 0):
                    self.pointer -= 1
            self.load_image()
            canvas.create_image(0, 0, image=self.current_image, anchor=NW)
            self.auto_paint()
            self.prevOval = None

    def paint(self, event):
        if self.prevOval is not None:
            canvas.delete(self.prevOval)
        python_blue = "#D9EA11"
        x1, y1 = event.x-3, event.y-3
        x2, y2 = event.x+3, event.y+3
        self.prevOval = canvas.create_oval(x1, y1, x2, y2, fill=python_blue)
        self.xys[self.pointer] = (event.x, event.y)


    def paint_blank(self, event):
        print('blank')
        if self.prevOval is not None:
            canvas.delete(self.prevOval)
        self.xys[self.pointer] = (-1, -1)


    def auto_paint(self):
        python_red = "#FF0000"
        print("{}---{}".format(str(self.pointer + 1), str(self.xys[self.pointer])))
        if self.xys[self.pointer][0] > 0:
            x1, y1 = self.xys[self.pointer][0]-3, self.xys[self.pointer][1]-3
            x2, y2 = self.xys[self.pointer][0]+3, self.xys[self.pointer][1]+3
            canvas.create_oval(x1, y1, x2, y2, fill=python_red)

    def load_directs_xys(self):
        xys = []
        directs = []
        for directory in self.dicts.keys():
            xys.append(self.dicts[directory])
            directs.append(directory)
        self.xys = xys
        self.directs = directs
        # self.pointer = 0

    # def update_coordinates(self, event):
        # with open(self.saved_dict, "w") as f:
            # for index, xy in enumerate(self.xys):
                # f.write(self.directs[index] + "," + str(xy[0]) + "," + str(xy[1]) + "\n")
        # print("xys saved to {}, current picture index is {}".format(self.saved_dict, str(self.pointer)))

    def update_coordinates(self, event):
        with open(self.pass_csv, "w") as f:
            for index, xy in enumerate(self.xys):
                f.write(self.directs[index] + "," + str(xy[0]) + "," + str(xy[1]) + "\n")
        print("xys saved to {}, current picture index is {}".format(self.pass_csv, str(self.pointer)))

    def load_image(self):
        # dir = './images/'
        dir = './pass_images/'
        img = ImageTk.PhotoImage(Image.open(os.path.join(dir, self.directs[self.pointer])))
        self.current_image = img

    def save_image(self, event):
        file_name = "pred_images/saved_test_{}.jpg".format(str(self.pointer))
        ImageGrab.grab(bbox=(10,10,1300,1300)).save(file_name)
        print("image saved to {}".format(file_name))


    def callback_get(self, event):
        if self.pointer < len(self.directs) - 1:
            canvas.delete("all")
            self.pointer = self.top_10[self.top_10_pointer % 10]
            self.top_10_pointer += 1
            self.load_image()
            canvas.create_image(0, 0, image=self.current_image, anchor=NW)
            self.auto_paint()
            self.prevOval = None



if __name__ == "__main__":
    pointer = input('Start with image pointer: ')
    only_invalid = input('Only invalid images (N/y)?')

    il = ImageLabel()
    il.pointer = int(pointer) if pointer != '' else il.pointer
    il.only_invalid = True if only_invalid.lower() == 'y' else False
    # il.load_dictionary_from_osdir("images", self.coordinates) # 1. load from original coodinates
    # il.load_dictionary_from_dict(il.saved_dict) #-------------- 2. load from corrected local dictionary
    il.load_dictionary_from_dict(il.pred_dict) # -----------------3. load from prediected dictionary
    # il.load_dictionary_from_dict(il.pass_csv) # -----------------4. load from pass_images dictionary

    canvas = Canvas(il.root)
    canvas.pack(expand=YES, fill=BOTH)
    il.load_image()
    canvas.create_image(0, 0, image=il.current_image, anchor=NW)
    il.auto_paint()
    canvas.bind("<Button-1>", il.paint)
    canvas.bind("<Button-3>", il.paint_blank)
    il.root.bind("<Left>", il.callback_previous)
    il.root.bind("<Right>", il.callback_next)
    il.root.bind("<space>", il.update_coordinates)
    il.root.bind("<Return>", il.save_image)
    il.root.bind("<Button-2>", il.callback_get)

    il.root.mainloop()
