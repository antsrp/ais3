import io

import PySimpleGUI as sg
import os.path

from PIL import Image, ImageTk


# First the window layout in 2 columns

def get_img_data(f, maxsize=(28, 28), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:  # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)


class MainLayout:
    def __init__(self, model):
        file_list_frame = [
            [
                sg.Text("Image file"),
                sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
                sg.FileBrowse(file_types=(("Image files", "*.png"), ("Image files", "*.jpg")), key="-IN-"),
            ],
            [
                sg.Button("Get an answer", key="-GAB-", disabled=True),
            ],
            [
                sg.Text("Answer of neural network:", key="-LABEL_ANSWER-", visible=False),
                sg.Text("", key="-ANSWER-", visible=False)
            ]
        ]

        column2 = [[sg.Image(key="-IMAGE-", size=(28, 28))]]
        layout = [
            [
                sg.Column(file_list_frame),
                sg.VSeperator(),
                sg.Column(column2)
            ]
        ]
        self.window = sg.Window("Neural network", layout)
        self.model = model

    def mainLoop(self):
        while True:
            event, values = self.window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "-FILE-":
                filename = values["-FILE-"]
                try:
                    image_data = get_img_data(filename)
                    # self.window["-IMAGE-"].update(filename=filename)
                    self.window["-IMAGE-"].update(data=image_data)
                    self.window["-GAB-"].Update(disabled=False)
                except:
                    sg.PopupOK('Error with opening a file!', title='Error')
            if event == "-GAB-":
                answer = self.model.predict(filename)
                self.window["-ANSWER-"].update(value=answer)
                self.window["-LABEL_ANSWER-"].Update(visible=True)
                self.window["-ANSWER-"].Update(visible=True)

        self.window.close()
        pass
