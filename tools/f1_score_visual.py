import os
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.image_files = []
        self.index = 0

        self.display = Label(root)
        self.display.pack()

        self.button_open = Button(root, text="Open Folder", command=self.open_folder)
        self.button_open.pack()

        self.root.bind('<Left>', lambda event: self.show_previous_image())
        self.root.bind('<Right>', lambda event: self.show_next_image())

    def open_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder = folder
            self.image_files = [file for file in os.listdir(folder) if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
            self.image_files.sort()
            self.index = 0
            self.show_image()

    def show_image(self):
        if self.image_files:
            image_path = os.path.join(self.folder, self.image_files[self.index])
            image = Image.open(image_path)
            # 放大两倍
            image = image.resize((image.width * 2, image.height * 2), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)

            self.display.config(image=photo)
            self.display.image = photo  # Keep a reference so it's not garbage collected
            self.root.title(f"Image Viewer ({self.index + 1}/{len(self.image_files)}) - {self.image_files[self.index]}")

    def show_previous_image(self):
        if self.image_files:
            self.index = (self.index - 1) % len(self.image_files)
            self.show_image()

    def show_next_image(self):
        if self.image_files:
            self.index = (self.index + 1) % len(self.image_files)
            self.show_image()

if __name__ == "__main__":
    root = Tk()
    app = ImageViewer(root)
    root.mainloop()
