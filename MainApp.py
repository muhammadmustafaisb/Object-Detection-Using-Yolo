import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

class YOLOModel:
    def __init__(self):
        self.weights_path = ""
        self.cfg_path = ""
        self.names_path = ""
        self.net = None
        self.classes = None
        self.output_layers = None

    def load_yolo(self):
        if self.weights_path and self.cfg_path and self.names_path:
            try:
                self.net = cv2.dnn.readNet(self.weights_path, self.cfg_path)
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                with open(self.names_path, "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
                messagebox.showinfo("YOLO", "YOLO model loaded successfully.")
            except Exception as e:
                messagebox.showerror("YOLO Error", f"Error loading YOLO: {e}")

    def apply_yolo(self, image):
        if not self.net:
            messagebox.showerror("YOLO Error", "YOLO model is not loaded. Please load the model first.")
            return image

        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

class ImageProcessor:
    def __init__(self):
        self.image_path = None
        self.image = None
        self.filter_mode = None
        self.detect_objects_flag = False

    def apply_blur(self, image):
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred

    def apply_sharpen(self, image):
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def apply_canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def apply_gaussian_blur(self, image):
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred

    def apply_median_blur(self, image):
        return cv2.medianBlur(image, 5)

    def apply_bilateral_filter(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def apply_sobel(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

    def apply_laplacian(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.cvtColor(cv2.convertScaleAbs(laplacian), cv2.COLOR_GRAY2RGB)

    def apply_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    def apply_filter(self, image):
        if self.filter_mode == "canny":
            return self.apply_canny(image)
        elif self.filter_mode == "sharpen":
            return self.apply_sharpen(image)
        elif self.filter_mode == "gaussian_blur":
            return self.apply_gaussian_blur(image)
        elif self.filter_mode == "median_blur":
            return self.apply_median_blur(image)
        elif self.filter_mode == "bilateral_filter":
            return self.apply_bilateral_filter(image)
        elif self.filter_mode == "sobel":
            return self.apply_sobel(image)
        elif self.filter_mode == "laplacian":
            return self.apply_laplacian(image)
        elif self.filter_mode == "threshold":
            return self.apply_threshold(image)
        return image

    def load_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "No image selected.")
            return

        self.image = cv2.imread(self.image_path)
        if self.image is None:
            messagebox.showerror("Error", "Failed to load image.")
            return

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.image

    def apply_filter_to_image(self, yolo_model):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        filtered_image = self.apply_filter(self.image)
        if self.detect_objects_flag:
            filtered_image = yolo_model.apply_yolo(filtered_image)

        return filtered_image


class VideoProcessor:
    def __init__(self):
        self.video_path = None
        self.video_capture = None
        self.running = False

    def process_video(self, apply_filter, apply_yolo, display_image):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = apply_filter(frame)
                frame = apply_yolo(frame)
                display_image(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            cap.release()
            self.stop_running()

    def show_live_video(self, apply_filter, apply_yolo, display_image):
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", "Could not open video capture device.")
            self.stop_running()
            return

        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = apply_filter(frame)
            frame = apply_yolo(frame)
            display_image(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_running()

    def stop_running(self):
        self.running = False

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        cv2.destroyAllWindows()

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV Function Explorer")

        screen_height = self.root.winfo_screenheight()
        screen_width = self.root.winfo_screenwidth()
        self.root.geometry("200x500+0+0")
        self.new_window = tk.Toplevel(root)
        self.new_window.geometry(f"600x{int(screen_height - 400)}+{int(screen_width/3)}+0")
        self.new_window.maxsize(800, 1000)
        self.new_window.title("Source")
        # self.new_window.panel = tk.Label(self.new_window)
        # self.new_window.panel.pack(padx=10, pady=10)
        self.new_window.panel = tk.Label(self.new_window)
        self.new_window.panel.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.new_window.resizable(False, False)

        self.yolo_model = YOLOModel()
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        self.thread = None

        self.create_buttons()
        self.new_window.protocol("WM_DELETE_WINDOW", self.close_windows)
        self.root.protocol("WM_DELETE_WINDOW", self.close_windows)
        self.create_scrollable_frame()

    def create_buttons(self):
        btn_frame = tk.Frame(self.new_window)
        btn_frame.pack(fill=tk.X, pady=10)

        btn_select_weights = tk.Button(btn_frame, text="Select Weights", command=self.select_weights)
        btn_select_weights.pack(side=tk.LEFT, padx=10)

        btn_select_cfg = tk.Button(btn_frame, text="Select CFG", command=self.select_cfg)
        btn_select_cfg.pack(side=tk.LEFT, padx=10)

        btn_select_names = tk.Button(btn_frame, text="Select Names", command=self.select_names)
        btn_select_names.pack(side=tk.LEFT, padx=10)

        btn_select_image = tk.Button(btn_frame, text="Select Image", command=self.select_image)
        btn_select_image.pack(side=tk.LEFT, padx=10)

        btn_select_video = tk.Button(btn_frame, text="Select Video", command=self.select_video)
        btn_select_video.pack(side=tk.LEFT, padx=10)

        btn_live_video = tk.Button(btn_frame, text="Live Video", command=self.toggle_live_video)
        btn_live_video.pack(side=tk.LEFT, padx=10)


    def on_resize(self, event):
        # Refresh the image on resize
        if self.image_processor.image is not None:
            self.display_image(self.image_processor.image)

    def close_windows(self):
        self.video_processor.stop_running()
        self.root.destroy()
        self.new_window.destroy()

    def select_weights(self):
        self.yolo_model.weights_path = filedialog.askopenfilename()
        self.yolo_model.load_yolo()

    def select_cfg(self):
        self.yolo_model.cfg_path = filedialog.askopenfilename()
        self.yolo_model.load_yolo()

    def select_names(self):
        self.yolo_model.names_path = filedialog.askopenfilename()
        self.yolo_model.load_yolo()

    def select_image(self):
        if self.video_processor.running:
            self.video_processor.stop_running()
        self.image_processor.image_path = filedialog.askopenfilename()
        if self.image_processor.image_path:
            self.load_image()

    def load_image(self):
        image = self.image_processor.load_image()
        if image is not None:
            if self.image_processor.detect_objects_flag:
                image = self.yolo_model.apply_yolo(image)
            self.display_image(image)

    def process_video(self):
        self.video_processor.running = True
        self.thread = threading.Thread(target=self.video_processor.process_video, args=(
            self.image_processor.apply_filter,
            self.yolo_model.apply_yolo,
            self.display_image
        ))
        self.thread.start()

    def select_video(self):
        if self.video_processor.running:
            self.video_processor.stop_running()

        self.video_processor.video_path = filedialog.askopenfilename()
        if self.video_processor.video_path:
            self.process_video()

    def toggle_live_video(self):
        if self.video_processor.running:
            self.video_processor.stop_running()
        else:
            self.start_live_video()

    def start_live_video(self):
        self.video_processor.running = True
        self.thread = threading.Thread(target=self.video_processor.show_live_video, args=(
            self.image_processor.apply_filter,
            self.yolo_model.apply_yolo,
            self.display_image
        ))
        self.thread.start()

    def toggle_detect_objects(self):
        self.image_processor.detect_objects_flag = not self.image_processor.detect_objects_flag
        if self.image_processor.image_path:
            self.apply_filter_to_image()

    def apply_filter_to_image(self):
        filtered_image = self.image_processor.apply_filter_to_image(self.yolo_model)
        if filtered_image is not None:
            self.display_image(filtered_image)

    def create_scrollable_frame(self):
        container = ttk.Frame(self.root)
        canvas = tk.Canvas(container, width=200, height=600)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack()
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.add_option_buttons(scrollable_frame)

    def add_option_buttons(self, frame):
        options = [
            ("Canny", lambda: self.set_filter_mode("canny")),
            ("Sharpen", lambda: self.set_filter_mode("sharpen")),
            ("Gaussian Blur", lambda: self.set_filter_mode("gaussian_blur")),
            ("Median Blur", lambda: self.set_filter_mode("median_blur")),
            ("Bilateral Filter", lambda: self.set_filter_mode("bilateral_filter")),
            ("Sobel", lambda: self.set_filter_mode("sobel")),
            ("Laplacian", lambda: self.set_filter_mode("laplacian")),
            ("Threshold", lambda: self.set_filter_mode("threshold")),
            ("Detect Objects", self.toggle_detect_objects),
            ("Reset", self.reset_image),
            ("Reset Filters", self.reset_filters),
        ]

        for option, command in options:
            btn = tk.Button(frame, text=option, command=command, width=15, pady=5)
            btn.pack(pady=5)

    def set_filter_mode(self, mode):
        self.image_processor.filter_mode = mode
        if self.image_processor.image_path:
            self.apply_filter_to_image()

    def reset_image(self):
        self.image_processor.image = cv2.imread(self.image_processor.image_path)
        self.image_processor.image = cv2.cvtColor(self.image_processor.image, cv2.COLOR_BGR2RGB)
        self.display_image(self.image_processor.image)

    def reset_filters(self):
        self.image_processor.filter_mode = None
        if self.image_processor.image_path:
            self.apply_filter_to_image()

    def display_image(self, img):
        display_width = self.new_window.panel.winfo_width()
        display_height = self.new_window.panel.winfo_height()

        img = Image.fromarray(img)

        img_width, img_height = img.size

        aspect_ratio = img_width / img_height

        if display_width / display_height > aspect_ratio:
            new_height = display_height
            new_width = int(aspect_ratio * new_height)
        else:
            new_width = display_width
            new_height = int(new_width / aspect_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)

        self.new_window.panel.imgtk = imgtk
        self.new_window.panel.config(image=imgtk)
        self.new_window.panel.image = imgtk

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
