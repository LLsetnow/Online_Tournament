import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# 用于对比数据集和预测结果

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("图片浏览器")
        self.zoom_factor = 2.5  # 初始放大倍数
        self.filter_options = ["无", "block", "bomb", "bridge", "cone", "crosswalk", "danger", "evil", "patient", "prop", "safety", "spy", "thief", "tumble"]

        self.folder1_path = ""
        self.folder2_path = ""
        self.current_index = 0
        self.filter_var = tk.StringVar(value="无")  # 用于存储选择的筛选条件

        self.label_image_name = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#212121", fg="white")
        self.title = tk.Label(root, text="左右键切换图片 上下键缩放", font=("Helvetica", 12, "bold"), bg="#212121", fg="white")
        self.filter_text = tk.Label(root, text="图像筛选", font=("Helvetica", 12, "bold"), bg="#212121",
                              fg="white")

        self.image_label1 = tk.Label(root, bg="#212121")
        self.image_label2 = tk.Label(root, bg="#212121")

        self.load_button1 = tk.Button(root, text="数据集", command=self.load_folder1, bg="#4CAF50", fg="white")
        self.load_button2 = tk.Button(root, text="预测结果", command=self.load_folder2, bg="#008CBA", fg="white")

        # 为下拉选择框增加选项"无"
        self.filter_options_combobox = ttk.Combobox(root, values=self.filter_options, textvariable=self.filter_var, state="readonly")
        self.filter_options_combobox.bind("<<ComboboxSelected>>", self.apply_filter)

        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Up>", self.zoom_in)
        self.root.bind("<Down>", self.zoom_out)

        # 设置窗口背景颜色
        self.root.config(bg="#212121")  # 使用十六进制颜色代码，这里设置为白色

        self.title.grid(row=0, columnspan=2, pady=10)

        self.load_button1.grid(row=1, column=0, padx=100, pady=10)
        self.load_button2.grid(row=1, column=1, padx=100, pady=10)

        # 使用新的下拉选择框
        self.filter_text.grid(row=2, columnspan=2, pady=10)
        self.filter_options_combobox.grid(row=3, columnspan=2, pady=10)
        self.image_label1.grid(row=4, column=0, padx=10, pady=10)
        self.image_label2.grid(row=4, column=1, padx=10, pady=10)

        self.label_image_name.grid(row=5, columnspan=2, pady=0)

    def load_folder1(self):
        folder1_path = filedialog.askdirectory()
        if folder1_path:
            self.folder1_path = folder1_path
            self.show_image1()

    def load_folder2(self):
        folder2_path = filedialog.askdirectory()
        if folder2_path:
            self.folder2_path = folder2_path
            self.show_image2()

    def show_image1(self):
        images1 = self.load_images(self.folder1_path)
        if images1 and 0 <= self.current_index < len(images1):
            img1 = ImageTk.PhotoImage(images1[self.current_index].resize(
                (int(images1[self.current_index].width * self.zoom_factor),
                 int(images1[self.current_index].height * self.zoom_factor))))
            self.image_label1.configure(image=img1)
            self.image_label1.image = img1
            self.update_image_name_label(self.folder1_path, self.current_index)

    def show_image2(self):
        images2 = self.load_images(self.folder2_path)
        if images2 and 0 <= self.current_index < len(images2):
            img2 = ImageTk.PhotoImage(images2[self.current_index].resize(
                (int(images2[self.current_index].width * self.zoom_factor),
                 int(images2[self.current_index].height * self.zoom_factor))))
            self.image_label2.configure(image=img2)
            self.image_label2.image = img2
            self.update_image_name_label(self.folder2_path, self.current_index)

    def update_image_name_label(self, folder_path, index):
        if folder_path:
            image_files = self.apply_filter_condition(folder_path)
            if 0 <= index < len(image_files):
                image_name = image_files[index]
                self.label_image_name.configure(text=f"图像名称：{image_name}")
            else:
                self.label_image_name.configure(text="")

    def apply_filter_condition(self, folder_path):
        selected_filter = self.filter_var.get()
        if selected_filter == "无":
            return [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        elif selected_filter:
            return [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg") and f.lower().startswith(selected_filter)]
        else:
            return []

    def load_images(self, folder_path):
        if folder_path:
            image_files = self.apply_filter_condition(folder_path)
            images = [Image.open(os.path.join(folder_path, f)) for f in image_files]
            return images

    def prev_image(self, event):
        images1 = self.load_images(self.folder1_path)
        images2 = self.load_images(self.folder2_path)
        if self.current_index > 0 and self.current_index < min(len(images1), len(images2)):
            self.current_index -= 1
            self.show_image1()
            self.show_image2()

    def next_image(self, event):
        images1 = self.load_images(self.folder1_path)
        images2 = self.load_images(self.folder2_path)
        if self.current_index < min(len(images1), len(images2)) - 1:
            self.current_index += 1
            self.show_image1()
            self.show_image2()

    def zoom_in(self, event):
        self.zoom_factor += 0.1
        self.show_image1()
        self.show_image2()

    def zoom_out(self, event):
        if self.zoom_factor > 0.1:
            self.zoom_factor -= 0.1
            self.show_image1()
            self.show_image2()

    def apply_filter(self, event):
        self.show_image1()
        self.show_image2()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1600x800")  # 设置初始窗口大小
    app = ImageViewer(root)
    root.mainloop()
